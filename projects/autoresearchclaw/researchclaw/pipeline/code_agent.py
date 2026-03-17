"""Advanced multi-phase code generation agent.

Phases
------
1. **Architecture Planning** — produce file structure and class hierarchy
   before writing any code.
2. **Execution-in-the-Loop** — generate code, run in sandbox, feed errors
   back to LLM for iterative repair.
3. **Solution Tree Search** — explore multiple candidate implementations,
   evaluate via sandbox, select the best (optional, higher cost).
4. **Multi-Agent Review** — coder-reviewer dialog for quality assurance.

Integration
-----------
``CodeAgent`` is instantiated inside ``_execute_code_generation`` in
``executor.py`` when ``config.experiment.code_agent.enabled`` is True.
It receives the same inputs (topic, exp_plan, metric, pkg_hint) and
returns ``CodeAgentResult`` with the generated files.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeAgentConfig:
    """Configuration for the advanced code generation agent.

    All phases are independently toggleable.  The default profile enables
    Phases 1 (architecture), 2 (exec-fix), and 4 (review), which gives
    a large quality boost at moderate extra cost.  Phase 3 (tree search)
    is opt-in because it multiplies both LLM and sandbox usage.
    """

    enabled: bool = True

    # Phase 1: Architecture planning
    architecture_planning: bool = True

    # Phase 2: Execution-in-the-loop
    exec_fix_max_iterations: int = 3
    exec_fix_timeout_sec: int = 60

    # Phase 3: Solution tree search (off by default)
    tree_search_enabled: bool = False
    tree_search_candidates: int = 3
    tree_search_max_depth: int = 2
    tree_search_eval_timeout_sec: int = 120

    # Phase 4: Multi-agent review dialog
    review_max_rounds: int = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SolutionNode:
    """One candidate solution in the search tree."""

    node_id: str
    files: dict[str, str]
    parent_id: str | None = None
    depth: int = 0
    # Evaluation
    runs_ok: bool = False
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    generation_method: str = "initial"


@dataclass
class CodeAgentResult:
    """Final output from the code agent."""

    files: dict[str, str]
    architecture_spec: str = ""
    validation_log: list[str] = field(default_factory=list)
    total_llm_calls: int = 0
    total_sandbox_runs: int = 0
    best_score: float = 0.0
    tree_nodes_explored: int = 0
    review_rounds: int = 0


# ---------------------------------------------------------------------------
# Sandbox protocol (structural typing — no import dependency)
# ---------------------------------------------------------------------------


class _SandboxResult(Protocol):  # pragma: no cover
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    metrics: dict[str, object]
    timed_out: bool


class _SandboxLike(Protocol):  # pragma: no cover
    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# CodeAgent
# ---------------------------------------------------------------------------


class CodeAgent:
    """Multi-phase code generation agent.

    Parameters
    ----------
    llm : LLMClient
        The language model client to use for code generation.
    prompts : PromptManager
        Manages prompt templates.
    config : CodeAgentConfig
        Agent configuration (toggles, limits, timeouts).
    stage_dir : Path
        Working directory for this stage (e.g. ``run_dir/stage-10``).
    sandbox_factory : callable, optional
        ``(ExperimentConfig, Path) -> SandboxLike``.  Required for
        Phases 2 and 3.
    experiment_config : ExperimentConfig, optional
        Passed to ``sandbox_factory`` when creating sandboxes.
    """

    def __init__(
        self,
        llm: Any,
        prompts: Any,
        config: CodeAgentConfig,
        stage_dir: Path,
        sandbox_factory: Any | None = None,
        experiment_config: Any | None = None,
    ) -> None:
        self._llm = llm
        self._pm = prompts
        self._cfg = config
        self._stage_dir = stage_dir
        self._sandbox_factory = sandbox_factory
        self._exp_config = experiment_config
        self._calls = 0
        self._runs = 0
        self._log: list[str] = []
        self._sandbox: _SandboxLike | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        max_tokens: int = 8192,
    ) -> CodeAgentResult:
        """Execute all enabled phases and return generated files."""
        t0 = time.time()
        self._log_event("CodeAgent.generate() started")

        # Phase 1: Architecture planning
        arch_spec = ""
        if self._cfg.architecture_planning:
            arch_spec = self._phase1_architecture(topic, exp_plan, metric)

        # Phase 2+3: Code generation + iterative improvement
        nodes_explored = 0
        if self._cfg.tree_search_enabled and self._sandbox_factory:
            best, nodes_explored = self._phase3_tree_search(
                topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
            )
        else:
            files = self._phase2_generate_and_fix(
                topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
            )
            best = SolutionNode(
                node_id="single", files=files, runs_ok=True, score=1.0,
            )

        # Phase 4: Review dialog
        review_rounds = 0
        if self._cfg.review_max_rounds > 0:
            best.files, review_rounds = self._phase4_review(
                best.files, topic, exp_plan, metric,
            )

        elapsed = time.time() - t0
        self._log_event(
            f"CodeAgent.generate() done in {elapsed:.1f}s — "
            f"{self._calls} LLM calls, {self._runs} sandbox runs"
        )

        return CodeAgentResult(
            files=best.files,
            architecture_spec=arch_spec,
            validation_log=list(self._log),
            total_llm_calls=self._calls,
            total_sandbox_runs=self._runs,
            best_score=best.score,
            tree_nodes_explored=nodes_explored,
            review_rounds=review_rounds,
        )

    # ── Phase 1: Architecture Planning ────────────────────────────────────

    def _phase1_architecture(
        self, topic: str, exp_plan: str, metric: str,
    ) -> str:
        """Generate an architecture spec before writing code."""
        self._log_event("Phase 1: Architecture planning")

        sp = self._pm.sub_prompt(
            "architecture_planning",
            topic=topic,
            exp_plan=exp_plan,
            metric=metric,
        )
        resp = self._chat(sp.system, sp.user, max_tokens=4096)

        # Extract YAML block from response
        arch_spec = resp.content
        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", arch_spec, re.DOTALL)
        if yaml_match:
            arch_spec = yaml_match.group(1).strip()

        self._log_event(f"  Architecture spec: {len(arch_spec)} chars")
        return arch_spec

    # ── Phase 2: Generate + Execution-in-the-Loop Fix ─────────────────────

    def _phase2_generate_and_fix(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        arch_spec: str,
        max_tokens: int,
    ) -> dict[str, str]:
        """Generate code, then iteratively fix via sandbox execution feedback."""
        self._log_event("Phase 2: Generate + exec-fix")

        # Initial generation (uses the existing code_generation prompt)
        files = self._generate_code(
            topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
        )
        if not files:
            self._log_event("  WARNING: empty generation, returning fallback")
            return files

        # Exec-fix loop (only when sandbox is available)
        if not self._sandbox_factory or self._cfg.exec_fix_max_iterations <= 0:
            return files

        for i in range(self._cfg.exec_fix_max_iterations):
            result = self._run_in_sandbox(files)
            if result.returncode == 0:
                self._log_event(f"  Exec-fix iter {i}: code runs OK")
                break

            self._log_event(
                f"  Exec-fix iter {i}: crashed (rc={result.returncode}), "
                f"stderr={len(result.stderr)} chars"
            )
            files = self._fix_runtime_error(files, result)

        return files

    def _generate_code(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        arch_spec: str,
        max_tokens: int,
    ) -> dict[str, str]:
        """Single code generation call with architecture spec injected."""
        # Inject architecture specification into the pkg_hint slot
        hint = pkg_hint
        if arch_spec:
            hint = (
                f"{pkg_hint}\n\n"
                "## ARCHITECTURE SPECIFICATION (follow this file and class structure)\n"
                f"{arch_spec}\n"
            )

        sp = self._pm.for_stage(
            "code_generation",
            topic=topic,
            metric=metric,
            pkg_hint=hint,
            exp_plan=exp_plan,
        )
        resp = self._chat(sp.system, sp.user, max_tokens=max_tokens)

        files = self._extract_files(resp.content)
        if not files and resp.content.strip():
            # Retry with higher token budget
            self._log_event("  Empty extraction, retrying with 32768 tokens")
            resp = self._chat(sp.system, sp.user, max_tokens=32768)
            files = self._extract_files(resp.content)

        return files

    def _fix_runtime_error(
        self, files: dict[str, str], result: Any,
    ) -> dict[str, str]:
        """Fix a runtime error using structured LLM feedback."""
        files_ctx = self._format_files(files)
        stderr_tail = (result.stderr or "")[-3000:]
        stdout_tail = "\n".join(
            (result.stdout or "").split("\n")[-50:]
        )

        sp = self._pm.sub_prompt(
            "code_exec_fix",
            stderr=stderr_tail or "(empty)",
            stdout_tail=stdout_tail or "(empty)",
            returncode=str(result.returncode),
            files_context=files_ctx,
        )
        resp = self._chat(sp.system, sp.user, max_tokens=16384)

        fixed = self._extract_files(resp.content)
        if fixed:
            merged = dict(files)
            merged.update(fixed)
            return merged
        return files

    # ── Phase 3: Solution Tree Search ─────────────────────────────────────

    def _phase3_tree_search(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        arch_spec: str,
        max_tokens: int,
    ) -> tuple[SolutionNode, int]:
        """Explore multiple candidate solutions via tree search."""
        self._log_event("Phase 3: Solution tree search")
        all_nodes: list[SolutionNode] = []

        # Generate initial candidates
        n_cand = self._cfg.tree_search_candidates
        for k in range(n_cand):
            self._log_event(f"  Generating candidate {k + 1}/{n_cand}")
            files = self._generate_code(
                topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
            )
            node = SolutionNode(
                node_id=f"gen-{k}",
                files=files,
                depth=0,
                generation_method="initial",
            )
            all_nodes.append(node)

        # Iterative evaluate-fix-branch loop
        for depth in range(self._cfg.tree_search_max_depth):
            # Evaluate unevaluated nodes
            for node in all_nodes:
                if node.returncode == -1:
                    self._evaluate_node(node, metric)

            # Sort by score
            all_nodes.sort(key=lambda n: n.score, reverse=True)

            self._log_event(
                f"  Depth {depth}: {len(all_nodes)} nodes, "
                f"best={all_nodes[0].node_id} score={all_nodes[0].score:.2f}"
            )

            # If best runs OK, we're done
            if all_nodes[0].runs_ok:
                break

            # Generate fix variants for top-2 crashing candidates
            new_nodes: list[SolutionNode] = []
            for node in all_nodes[:2]:
                if not node.runs_ok:
                    fixed_files = self._fix_runtime_error(
                        node.files,
                        _SimpleResult(
                            returncode=node.returncode,
                            stdout=node.stdout,
                            stderr=node.stderr,
                        ),
                    )
                    new_node = SolutionNode(
                        node_id=f"{node.node_id}-fix{depth}",
                        files=fixed_files,
                        parent_id=node.node_id,
                        depth=depth + 1,
                        generation_method="fix",
                    )
                    new_nodes.append(new_node)

            all_nodes.extend(new_nodes)

        # Final evaluation of any remaining unevaluated nodes
        for node in all_nodes:
            if node.returncode == -1:
                self._evaluate_node(node, metric)

        all_nodes.sort(key=lambda n: n.score, reverse=True)
        best = all_nodes[0]
        self._log_event(
            f"  Tree search complete: best={best.node_id} "
            f"score={best.score:.2f}, explored {len(all_nodes)} nodes"
        )

        return best, len(all_nodes)

    def _evaluate_node(self, node: SolutionNode, metric_key: str) -> None:
        """Run a node's code in sandbox and update its score."""
        if not node.files:
            node.score = 0.0
            return

        result = self._run_in_sandbox(
            node.files,
            timeout_sec=self._cfg.tree_search_eval_timeout_sec,
        )
        node.returncode = result.returncode
        node.stdout = result.stdout
        node.stderr = result.stderr
        node.runs_ok = result.returncode == 0
        node.metrics = dict(result.metrics) if result.metrics else {}
        node.score = self._score_node(node, metric_key)

    @staticmethod
    def _score_node(node: SolutionNode, metric_key: str) -> float:
        """Score a solution node based on execution results."""
        score = 0.0
        if node.runs_ok:
            score += 1.0
        if node.stdout and len(node.stdout) > 100:
            score += 0.3  # produces meaningful output
        if node.metrics:
            score += 0.5
            if metric_key in node.metrics:
                score += 0.5
        if node.stderr and "Error" in node.stderr:
            score -= 0.2
        return max(score, 0.0)

    # ── Phase 4: Multi-Agent Review Dialog ────────────────────────────────

    def _phase4_review(
        self,
        files: dict[str, str],
        topic: str,
        exp_plan: str,
        metric: str,
    ) -> tuple[dict[str, str], int]:
        """Reviewer agent examines code; coder fixes critical issues."""
        self._log_event("Phase 4: Review dialog")

        rounds = 0
        for r in range(self._cfg.review_max_rounds):
            rounds += 1
            files_ctx = self._format_files(files)

            sp = self._pm.sub_prompt(
                "code_reviewer",
                topic=topic,
                exp_plan=exp_plan,
                metric=metric,
                files_context=files_ctx,
            )
            resp = self._chat(sp.system, sp.user, max_tokens=4096)

            review = self._parse_json(resp.content)
            if not isinstance(review, dict) or not review:
                self._log_event(
                    f"  Review round {r + 1}: could not parse JSON, skipping"
                )
                break

            verdict = review.get("verdict", "APPROVE")
            score = review.get("score", 10)
            critical = review.get("critical_issues", [])

            self._log_event(
                f"  Review round {r + 1}: verdict={verdict}, score={score}, "
                f"critical_issues={len(critical)}"
            )

            if verdict == "APPROVE" or not critical:
                break

            # Fix critical issues using the code_generation system prompt
            fix_prompt = (
                "A code reviewer found these critical issues in your experiment code.\n"
                "Fix ALL of them while preserving the experiment design.\n\n"
                "## Critical Issues\n"
                + "\n".join(f"- {issue}" for issue in critical)
                + f"\n\n## Current Code\n{files_ctx}\n\n"
                "Output ALL files in ```filename:xxx.py``` format, "
                "including unchanged files."
            )
            sys_prompt = self._pm.system("code_generation")
            fix_resp = self._chat(sys_prompt, fix_prompt, max_tokens=16384)

            fixed = self._extract_files(fix_resp.content)
            if fixed:
                files = dict(files)
                files.update(fixed)

        return files, rounds

    # ── Helpers ────────────────────────────────────────────────────────────

    def _chat(self, system: str, user: str, max_tokens: int = 8192) -> Any:
        """Make an LLM call and track count."""
        self._calls += 1
        messages = [{"role": "user", "content": user}]
        return self._llm.chat(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
        )

    def _get_or_create_sandbox(self) -> _SandboxLike:
        """Lazily create a single sandbox instance for all validation runs."""
        if self._sandbox is None:
            sandbox_dir = self._stage_dir / "agent_sandbox"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            self._sandbox = self._sandbox_factory(
                self._exp_config, sandbox_dir,
            )
        return self._sandbox

    def _run_in_sandbox(
        self,
        files: dict[str, str],
        timeout_sec: int | None = None,
    ) -> Any:
        """Write files to a temp directory and run in sandbox."""
        if not self._sandbox_factory:
            raise RuntimeError("No sandbox factory configured")

        self._runs += 1
        timeout = timeout_sec or self._cfg.exec_fix_timeout_sec

        # Write files to a numbered attempt directory
        run_dir = self._stage_dir / "agent_runs" / f"attempt_{self._runs:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for fname, code in files.items():
            fpath = run_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")

        # Run using the sandbox
        sandbox = self._get_or_create_sandbox()
        try:
            result = sandbox.run_project(run_dir, timeout_sec=timeout)
        except Exception as exc:
            self._log_event(f"  Sandbox run failed: {exc}")
            result = _SimpleResult(
                returncode=1,
                stdout="",
                stderr=f"Sandbox exception: {exc}",
            )

        return result

    def _extract_files(self, content: str) -> dict[str, str]:
        """Extract multi-file code blocks from LLM output."""
        # Local import to avoid circular dependency with executor.py
        from researchclaw.pipeline.executor import _extract_multi_file_blocks

        return _extract_multi_file_blocks(content)

    @staticmethod
    def _format_files(files: dict[str, str]) -> str:
        """Format files for inclusion in a prompt."""
        parts = []
        for fname in sorted(files):
            parts.append(f"```filename:{fname}\n{files[fname]}\n```")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Best-effort JSON extraction from LLM response."""
        # Direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
        # ```json``` fenced block
        m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
        # First {...} object
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _log_event(self, msg: str) -> None:
        """Log to both Python logger and the internal validation log."""
        logger.info("[CodeAgent] %s", msg)
        self._log.append(msg)


# ---------------------------------------------------------------------------
# Lightweight result stand-in for error plumbing
# ---------------------------------------------------------------------------


@dataclass
class _SimpleResult:
    """Minimal sandbox result for internal error plumbing."""

    returncode: int = 1
    stdout: str = ""
    stderr: str = ""
    elapsed_sec: float = 0.0
    metrics: dict[str, object] = field(default_factory=dict)
    timed_out: bool = False
