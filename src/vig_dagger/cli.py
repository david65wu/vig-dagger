"""Command line: ``vig-dagger <command> -c <experiment.yaml> [--set key=value ...]``.

  check                  preflight: GPUs, rendering, ffmpeg, disk, Hugging Face and W&B access (run before `run`)
  run                    the whole experiment (every part of every round) as one queue of jobs; jobs start when
                         their inputs exist and the machine has room. Rerun after an interruption: finished jobs are
                         skipped, still-running ones adopted. --only REGEX restricts it to matching jobs (and what they
                         need); --dry-run lists the jobs.
  status                 is the queue alive; failed and running jobs with their progress; milestones
  stop                   stop the queue and all its jobs (they resume where they stopped on the next `run`);
                         --queue-only: stop only the queue, the jobs keep running and the next `run` takes them over
  analyze --round R      recompute a round's DAgger metrics (results/ and W&B)
  summarize              recompute the evaluation summary (results/ and W&B)

Data and checkpoints:

  download-demos / download-policy   from the Hugging Face repos in hub.*
  upload {demos,policy,dagger}       to the Hugging Face repos in hub.*
  import-demos FILE.h5 ...           use existing planner trajectory files as the raw demos (part order = argument order;
                                     --extra: as the extra planner demos)
  import-policy CKPT [--thresholds thresholds.json]   use an existing MoE checkpoint as round moe's driver
  show-config                        print the resolved configuration
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from vig_dagger import config as config_module


def _select(jobs, pattern: str):
    """Jobs matching the pattern, plus every job they depend on."""
    by_name = {j.name: j for j in jobs}
    keep, stack = set(), [j.name for j in jobs if re.search(pattern, j.name)]
    while stack:
        name = stack.pop()
        if name not in keep:
            keep.add(name)
            stack += by_name[name].deps
    return [j for j in jobs if j.name in keep]


def main() -> None:
    parser = argparse.ArgumentParser(prog="vig-dagger", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command")
    parser.add_argument("items", nargs="*", help="upload targets, import files or checkpoint")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set dagger.num_demos=200 (repeatable)")
    parser.add_argument("--only", help="run: only jobs whose name matches this regular expression (and their inputs)")
    parser.add_argument("--dry-run", action="store_true", help="run: list the jobs instead of running them")
    parser.add_argument("--queue-only", action="store_true", help="stop: only the queue; running jobs continue")
    parser.add_argument("--round", help="analyze: the round")
    parser.add_argument("--extra", action="store_true", help="import-demos: the extra planner demos")
    parser.add_argument("--thresholds", type=Path, help="import-policy: the checkpoint's thresholds.json")
    args = parser.parse_intermixed_args()

    cfg = config_module.load(args.config, args.overrides)
    from vig_dagger import hub, plan, steps

    command = args.command
    if command == "show-config":
        print(yaml.safe_dump(cfg, sort_keys=False))
    elif command == "run":
        from vig_dagger.config import Paths
        from vig_dagger.jobs import Queue, lock

        config_path = plan.write_config(cfg)
        jobs = plan.build(cfg, config_path)
        if args.only:
            jobs = _select(jobs, args.only)
        if args.dry_run:
            for j in jobs:
                state = "done" if j.done is not None and j.done() else ""
                print(f"{j.priority:3d} {j.kind:5s} {j.name:60s} {state:5s} <- {', '.join(j.deps[:4])}{' ...' if len(j.deps) > 4 else ''}")
            return
        lock(Paths.of(cfg).root)
        done, stuck = Queue(jobs, cfg["resources"], Paths.of(cfg).logs()).run()
        print(f"[queue] finished: {len(done)} of {len(jobs)} jobs done" + (f"; failed or blocked: {sorted(stuck)}" if stuck else ""))
        if stuck:
            raise SystemExit(1)
    elif command == "check":
        from vig_dagger import ops

        if not ops.check(cfg):
            raise SystemExit(1)
    elif command == "status":
        from vig_dagger import ops

        ops.status(cfg)
    elif command == "stop":
        from vig_dagger import ops

        ops.stop(cfg, queue_only=args.queue_only)
    elif command == "analyze":
        steps.analyze(cfg, args.round)
    elif command == "summarize":
        steps.summarize(cfg)
    elif command == "download-demos":
        hub.download(cfg, "demos")
    elif command == "download-policy":
        hub.download(cfg, "policy")
    elif command == "upload":
        for what in args.items or ["demos", "policy"]:
            hub.upload(cfg, what)
    elif command == "import-demos":
        hub.import_demos(cfg, [Path(p) for p in args.items], extra=args.extra)
    elif command == "import-policy":
        hub.import_policy(cfg, Path(args.items[0]), args.thresholds)
    else:
        parser.error(f"unknown command {command!r}; see vig-dagger --help")


if __name__ == "__main__":
    main()
