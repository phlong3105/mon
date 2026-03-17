"""FigureAgent Orchestrator — coordinates the five sub-agents.

Flow: Planner → CodeGen → Renderer → Critic (→ retry CodeGen if failed)
     → Integrator

Produces a ``FigurePlan`` consumed by paper draft and export stages.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchclaw.agents.base import AgentOrchestrator
from researchclaw.agents.figure_agent.codegen import CodeGenAgent
from researchclaw.agents.figure_agent.critic import CriticAgent
from researchclaw.agents.figure_agent.integrator import IntegratorAgent
from researchclaw.agents.figure_agent.planner import PlannerAgent
from researchclaw.agents.figure_agent.renderer import RendererAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureAgentConfig:
    """Configuration for the FigureAgent system."""

    enabled: bool = True
    # Planner
    min_figures: int = 3
    max_figures: int = 8
    # Orchestrator
    max_iterations: int = 3   # max CodeGen→Renderer→Critic retry loops
    # Renderer
    render_timeout_sec: int = 30
    # Critic
    strict_mode: bool = False  # if True, any issue = fail
    # Output
    dpi: int = 300


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------


@dataclass
class FigurePlan:
    """Final output from the FigureAgent system.

    Consumed by:
    - Paper draft stage (figure_descriptions for writing prompt)
    - Paper export stage (manifest for LaTeX figure embedding)
    - Charts directory (scripts + rendered images)
    """

    # Figure manifest (list of figure metadata dicts)
    manifest: list[dict[str, Any]] = field(default_factory=list)

    # Generated references
    markdown_refs: str = ""
    figure_descriptions: str = ""

    # Paths
    output_dir: str = ""
    manifest_path: str = ""

    # Stats
    figure_count: int = 0
    passed_count: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "manifest": self.manifest,
            "markdown_refs": self.markdown_refs,
            "figure_descriptions": self.figure_descriptions,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "figure_count": self.figure_count,
            "passed_count": self.passed_count,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "elapsed_sec": self.elapsed_sec,
        }

    def get_chart_files(self) -> list[str]:
        """Return list of chart filenames from manifest."""
        return [
            Path(entry["file_path"]).name
            for entry in self.manifest
            if entry.get("file_path")
        ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FigureOrchestrator(AgentOrchestrator):
    """Coordinates Planner → CodeGen → Renderer → Critic → Integrator."""

    def __init__(
        self,
        llm: Any,
        config: FigureAgentConfig | None = None,
        *,
        stage_dir: Path | None = None,
    ) -> None:
        cfg = config or FigureAgentConfig()
        super().__init__(llm, max_iterations=cfg.max_iterations)

        self._config = cfg
        self._stage_dir = stage_dir

        # Initialize sub-agents
        self._planner = PlannerAgent(
            llm,
            min_figures=cfg.min_figures,
            max_figures=cfg.max_figures,
        )
        self._codegen = CodeGenAgent(llm)
        self._renderer = RendererAgent(
            llm,
            timeout_sec=cfg.render_timeout_sec,
        )
        self._critic = CriticAgent(
            llm,
            strict_mode=cfg.strict_mode,
        )
        self._integrator = IntegratorAgent(llm)

    def _save_artifact(self, name: str, data: Any) -> None:
        """Save intermediate artifact to stage directory."""
        if self._stage_dir is None:
            return
        self._stage_dir.mkdir(parents=True, exist_ok=True)
        path = self._stage_dir / name
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    def orchestrate(self, context: dict[str, Any]) -> FigurePlan:
        """Run the full figure generation pipeline.

        Context keys:
            experiment_results (dict): Parsed results.json
            condition_summaries (dict): Per-condition aggregated stats
            metrics_summary (dict): Per-metric aggregated stats
            metric_key (str): Primary metric name
            topic (str): Research topic
            hypothesis (str): Research hypothesis
            output_dir (str|Path): Directory for chart output
        """
        t0 = time.monotonic()
        topic = context.get("topic", "")
        output_dir = Path(context.get("output_dir", "charts"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("FigureAgent starting for: %s", topic[:80])

        plan = FigurePlan(output_dir=str(output_dir))

        # ── Phase 1: Plan ────────────────────────────────────────────
        self.logger.info("Phase 1: Planning figures")
        plan_result = self._planner.execute({
            "experiment_results": context.get("experiment_results", {}),
            "topic": topic,
            "hypothesis": context.get("hypothesis", ""),
            "conditions": context.get("conditions", []),
            "metric_key": context.get("metric_key", "primary_metric"),
            "metrics_summary": context.get("metrics_summary", {}),
            "condition_summaries": context.get("condition_summaries", {}),
        })
        self._accumulate(plan_result)

        if not plan_result.success:
            self.logger.warning("Planning failed: %s", plan_result.error)
            plan.elapsed_sec = time.monotonic() - t0
            plan.total_llm_calls = self.total_llm_calls
            plan.total_tokens = self.total_tokens
            return plan

        figures = plan_result.data.get("figures", [])
        self._save_artifact("figure_plan.json", figures)
        self.logger.info("Planned %d figures", len(figures))

        # ── Phase 2+3+4: CodeGen → Render → Critic (with retry) ─────
        critic_feedback: list[dict[str, Any]] = []
        final_rendered: list[dict[str, Any]] = []
        final_scripts: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            self.logger.info(
                "Phase 2: CodeGen (iteration %d/%d)",
                iteration + 1, self.max_iterations,
            )

            # CodeGen
            codegen_result = self._codegen.execute({
                "figures": figures,
                "experiment_results": context.get("experiment_results", {}),
                "condition_summaries": context.get("condition_summaries", {}),
                "metrics_summary": context.get("metrics_summary", {}),
                "metric_key": context.get("metric_key", "primary_metric"),
                "output_dir": str(output_dir),
                "critic_feedback": critic_feedback,
            })
            self._accumulate(codegen_result)

            if not codegen_result.success:
                self.logger.warning("CodeGen failed: %s", codegen_result.error)
                continue

            scripts = codegen_result.data.get("scripts", [])
            final_scripts = scripts
            self._save_artifact(f"scripts_{iteration}.json", [
                {k: v for k, v in s.items() if k != "script"}
                for s in scripts
            ])

            # Render
            self.logger.info(
                "Phase 3: Rendering (iteration %d/%d)",
                iteration + 1, self.max_iterations,
            )
            render_result = self._renderer.execute({
                "scripts": scripts,
                "output_dir": str(output_dir),
            })
            self._accumulate(render_result)

            if not render_result.success:
                self.logger.warning("Rendering failed: %s", render_result.error)
                continue

            rendered = render_result.data.get("rendered", [])
            final_rendered = rendered

            # Critic
            self.logger.info(
                "Phase 4: Critic review (iteration %d/%d)",
                iteration + 1, self.max_iterations,
            )
            critic_result = self._critic.execute({
                "rendered": rendered,
                "scripts": scripts,
                "condition_summaries": context.get("condition_summaries", {}),
                "metrics_summary": context.get("metrics_summary", {}),
                "metric_key": context.get("metric_key", "primary_metric"),
            })
            self._accumulate(critic_result)

            reviews = critic_result.data.get("reviews", [])
            all_passed = critic_result.data.get("all_passed", False)
            self._save_artifact(f"reviews_{iteration}.json", reviews)

            if all_passed:
                self.logger.info(
                    "All figures passed review on iteration %d", iteration + 1
                )
                break

            # Collect feedback for failed figures (for next iteration)
            critic_feedback = [
                r for r in reviews if not r.get("passed")
            ]

            # Only retry figures that failed
            failed_ids = {r["figure_id"] for r in critic_feedback}
            figures = [f for f in figures if f.get("figure_id") in failed_ids]

            self.logger.warning(
                "Critic: %d/%d figures need revision",
                len(failed_ids), len(rendered),
            )

        # ── Phase 5: Integrate ───────────────────────────────────────
        self.logger.info("Phase 5: Integrating figures into paper")
        integrate_result = self._integrator.execute({
            "rendered": final_rendered,
            "topic": topic,
            "output_dir": str(output_dir),
        })
        self._accumulate(integrate_result)

        # ── Finalize ─────────────────────────────────────────────────
        plan.manifest = integrate_result.data.get("manifest", [])
        plan.markdown_refs = integrate_result.data.get("markdown_refs", "")
        plan.figure_descriptions = integrate_result.data.get("figure_descriptions", "")
        plan.manifest_path = integrate_result.data.get("manifest_path", "")
        plan.figure_count = integrate_result.data.get("figure_count", 0)
        plan.passed_count = sum(
            1 for r in final_rendered if r.get("success")
        )
        plan.total_llm_calls = self.total_llm_calls
        plan.total_tokens = self.total_tokens
        plan.elapsed_sec = time.monotonic() - t0

        # Save final plan
        self._save_artifact("figure_plan_final.json", plan.to_dict())

        self.logger.info(
            "FigureAgent complete: %d figures, %d passed review, "
            "%d LLM calls, %.1fs",
            plan.figure_count,
            plan.passed_count,
            plan.total_llm_calls,
            plan.elapsed_sec,
        )

        return plan
