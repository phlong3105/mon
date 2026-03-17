"""ResearchClaw CLI — run the 23-stage autonomous research pipeline."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import cast

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.health import print_doctor_report, run_doctor, write_doctor_report


def _generate_run_id(topic: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:6]
    return f"rc-{ts}-{topic_hash}"


def cmd_run(args: argparse.Namespace) -> int:
    config_path = Path(cast(str, args.config))
    topic = cast(str | None, args.topic)
    output = cast(str | None, args.output)
    from_stage_name = cast(str | None, args.from_stage)
    auto_approve = cast(bool, args.auto_approve)
    skip_preflight = cast(bool, args.skip_preflight)
    resume = cast(bool, args.resume)
    skip_noncritical = cast(bool, args.skip_noncritical_stage)

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    kb_root_path = None
    config = RCConfig.load(config_path, check_paths=False)

    if topic:
        import dataclasses

        new_research = dataclasses.replace(config.research, topic=topic)
        config = dataclasses.replace(config, research=new_research)

    # --- LLM Preflight ---
    if not skip_preflight:
        from researchclaw.llm import create_llm_client

        client = create_llm_client(config)
        print("Preflight check...", end=" ", flush=True)
        ok, msg = client.preflight()
        if ok:
            print(msg)
        else:
            print(f"FAILED — {msg}", file=sys.stderr)
            return 1

    run_id = _generate_run_id(config.research.topic)
    run_dir = Path(output or f"artifacts/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.knowledge_base.root:
        kb_root_path = Path(config.knowledge_base.root)
        kb_root_path.mkdir(parents=True, exist_ok=True)

    adapters = AdapterBundle()

    from researchclaw.pipeline.runner import execute_pipeline, read_checkpoint
    from researchclaw.pipeline.stages import Stage

    # --- Determine start stage ---
    from_stage = Stage.TOPIC_INIT
    if from_stage_name:
        from_stage = Stage[from_stage_name.upper()]
    elif resume:
        resumed = read_checkpoint(run_dir)
        if resumed is not None:
            from_stage = resumed
            print(f"Resuming from checkpoint: Stage {int(from_stage)}: {from_stage.name}")

    print(f"ResearchClaw v0.1.0 — Starting pipeline")
    print(f"  Run ID:  {run_id}")
    print(f"  Topic:   {config.research.topic}")
    print(f"  Output:  {run_dir}")
    print(f"  Mode:    {config.experiment.mode}")
    print(f"  From:    Stage {int(from_stage)}: {from_stage.name}")
    print()

    results = execute_pipeline(
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        adapters=adapters,
        from_stage=from_stage,
        auto_approve_gates=auto_approve,
        skip_noncritical=skip_noncritical,
        kb_root=kb_root_path,
    )

    done = sum(1 for r in results if r.status.value == "done")
    failed = sum(1 for r in results if r.status.value == "failed")
    print(f"\nPipeline complete: {done}/{len(results)} stages done, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from researchclaw.config import validate_config
    import yaml

    config_path = Path(cast(str, args.config))
    no_check_paths = cast(bool, args.no_check_paths)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open(encoding="utf-8") as f:
        loaded = cast(object, yaml.safe_load(f))

    if loaded is None:
        data: dict[str, object] = {}
    elif isinstance(loaded, dict):
        loaded_map = cast(Mapping[object, object], loaded)
        data = {str(key): value for key, value in loaded_map.items()}
    else:
        print("Config validation FAILED:")
        print("  Error: Config root must be a mapping")
        return 1

    result = validate_config(data, check_paths=not no_check_paths)
    if result.ok:
        print("Config validation passed")
        for w in result.warnings:
            print(f"  Warning: {w}")
        return 0
    else:
        print("Config validation FAILED:")
        for e in result.errors:
            print(f"  Error: {e}")
        return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    config_path = Path(cast(str, args.config))
    output = cast(str | None, args.output)

    report = run_doctor(config_path)
    print_doctor_report(report)
    if output:
        write_doctor_report(report, Path(output))
    return 0 if report.overall == "pass" else 1

def cmd_report(args: argparse.Namespace) -> int:
    from researchclaw.report import generate_report, write_report

    run_dir = Path(cast(str, args.run_dir))
    output = cast(str | None, args.output)

    try:
        report = generate_report(run_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(report)
    if output:
        write_report(run_dir, Path(output))
        print(f"\nReport written to {output}")
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="researchclaw",
        description="ResearchClaw — Autonomous Research Pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the 23-stage research pipeline")
    _ = run_p.add_argument("--topic", "-t", help="Override research topic")
    _ = run_p.add_argument(
        "--config", "-c", default="config.yaml", help="Config file path"
    )
    _ = run_p.add_argument("--output", "-o", help="Output directory")
    _ = run_p.add_argument(
        "--from-stage", help="Start from a specific stage (e.g. PAPER_OUTLINE)"
    )
    _ = run_p.add_argument(
        "--auto-approve", action="store_true", help="Auto-approve gate stages"
    )
    _ = run_p.add_argument(
        "--skip-preflight", action="store_true", help="Skip LLM preflight check"
    )
    _ = run_p.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    _ = run_p.add_argument(
        "--skip-noncritical-stage", action="store_true",
        help="Skip noncritical stages on failure instead of aborting"
    )
    val_p = sub.add_parser("validate", help="Validate config file")
    _ = val_p.add_argument(
        "--config", "-c", default="config.yaml", help="Config file path"
    )
    _ = val_p.add_argument(
        "--no-check-paths", action="store_true", help="Skip path existence checks"
    )

    doc_p = sub.add_parser("doctor", help="Check environment and configuration health")
    _ = doc_p.add_argument(
        "--config", "-c", default="config.yaml", help="Config file path"
    )
    _ = doc_p.add_argument("--output", "-o", help="Write JSON report to file")

    rpt_p = sub.add_parser("report", help="Generate human-readable run report")
    _ = rpt_p.add_argument(
        "--run-dir", required=True, help="Path to run artifacts directory"
    )
    _ = rpt_p.add_argument("--output", "-o", help="Write report to file")
    args = parser.parse_args(argv)

    command = cast(str | None, args.command)

    if command == "run":
        return cmd_run(args)
    elif command == "validate":
        return cmd_validate(args)
    elif command == "doctor":
        return cmd_doctor(args)
    elif command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
