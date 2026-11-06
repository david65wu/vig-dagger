"""Command line: ``vig-dagger <command> -c <experiment.yaml> [--set key=value ...]``.

  check                  preflight: GPUs, rendering, ffmpeg, disk, Hugging Face and W&B access (run before `run`)
  run                    the whole experiment as one queue of jobs; jobs start when their inputs exist and the
                         machine has room. Rerun after an interruption: finished jobs are skipped, still-running ones
                         adopted. --tasks N ...: only these numbered tasks (several machines, see `tasks`);
                         --only REGEX: only matching jobs (and what they need); --dry-run lists the jobs.
  tasks                  the numbered tasks (what each needs and who uses its outputs), as in TASKS.md
  pack --tasks N ... [--for M] [--out FILE]   one archive of these tasks' outputs (only what task M needs of them);
                         checkpoints weights-only
  unpack FILE            put an archive's files in place in this work directory
  status                 is the queue alive; failed and running jobs with their progress; milestones
  stop                   stop the queue and all its jobs (they resume where they stopped on the next `run`);
                         --queue-only: stop only the queue, the jobs keep running and the next `run` takes them over
  report                 write results/REPORT.md, results/summary.* and results/figures/ from the results so far
  analyze --collection C recompute one collection's metrics (all analyzed collections without --collection)

Data and checkpoints:

  download-demos                       the planner demos from hub.dataset_repo
  download-moe --seed S                a MoE from its repo in moe.download
  upload {demos,dagger}                to hub.dataset_repo
  import-demos FILE.h5 ...             use existing planner trajectory files as the raw demos (in argument order)
  import-moe CKPT --seed S             use an existing MoE checkpoint as the MoE of seed S
  show-config                          print the resolved configuration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vig_dagger import config as config_module


def main() -> None:
    parser = argparse.ArgumentParser(prog="vig-dagger", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command")
    parser.add_argument("items", nargs="*", help="upload targets, import files, an archive or a checkpoint")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set dagger.num_demos=200 (repeatable)")
    parser.add_argument("--tasks", nargs="+", help="run / pack: task numbers or ranges, e.g. --tasks 1 4-9")
    parser.add_argument("--for", dest="reader", type=int, help="pack: only what this task needs")
    parser.add_argument("--out", type=Path, help="pack: the archive (default <work_dir>/packs/...)")
    parser.add_argument("--only", help="run: only jobs whose name matches this regular expression (and their inputs)")
    parser.add_argument("--dry-run", action="store_true", help="run: list the jobs instead of running them")
    parser.add_argument("--queue-only", action="store_true", help="stop: only the queue; running jobs continue")
    parser.add_argument("--collection", help="analyze: the collection id, e.g. sd0_vig_p90_plain")
    parser.add_argument("--seed", type=int, help="download-moe, import-moe: the MoE seed")
    args = parser.parse_intermixed_args()

    cfg = config_module.load(args.config, args.overrides)
    from vig_dagger import hub, plan, steps
    from vig_dagger import tasks as task_list
    from vig_dagger.config import Collection, Paths

    command = args.command
    if command == "show-config":
        print(yaml.safe_dump(cfg, sort_keys=False))
    elif command == "run":
        from vig_dagger.jobs import Queue, lock

        config_path = plan.write_config(cfg)
        jobs = plan.build(cfg, config_path)
        if args.tasks:  # several machines, tasks assigned by hand: these tasks and what they need
            names = task_list.numbered(cfg, jobs)
            selected = task_list.parse(args.tasks, names)
            problems = task_list.check_inputs(cfg, jobs, selected)
            if problems:
                raise SystemExit("cannot run yet; copy these outputs here first (`vig-dagger pack` on the machine that "
                                 "made them, `vig-dagger unpack` here):\n  " + "\n  ".join(problems))
            jobs = task_list.select_jobs(cfg, jobs, selected)
            print(f"[tasks] {', '.join(f'{names.index(t) + 1} ({t})' for t in selected)}", flush=True)
        if args.only:
            jobs = plan.select(jobs, args.only)
        if args.dry_run:
            for j in jobs:
                state = "done" if j.done is not None and j.done() else ""
                print(f"{j.priority:3d} {j.kind:5s} {j.name:60s} {state:5s} <- {', '.join(j.deps[:4])}{' ...' if len(j.deps) > 4 else ''}")
            return
        lock(Paths.of(cfg).root)
        done, stuck = Queue(jobs, cfg["resources"], Paths.of(cfg).logs()).run()
        print(f"[queue] finished: {len(done)} of {len(jobs)} jobs done" + (f"; failed or blocked: {sorted(stuck)}" if stuck else ""))
        if args.tasks and not stuck:
            all_jobs = plan.build(cfg, config_path)
            names = task_list.numbered(cfg, all_jobs)
            by_name = {j.name: j for j in all_jobs}
            ran = task_list.parse(args.tasks, names)
            users = sorted({names.index(t) + 1 for t in names if set(task_list.needs(by_name, t)) & set(ran)} - {names.index(t) + 1 for t in ran})
            if users:
                print(f"[tasks] done; tasks {task_list._ranges(users)} use these outputs (`vig-dagger pack --tasks ... --for N`)")
        if stuck:
            raise SystemExit(1)
    elif command == "tasks":
        jobs = plan.build(cfg, Paths.of(cfg).config())
        print(task_list.markdown(cfg, jobs))
    elif command == "pack":
        if not args.tasks:
            parser.error("pack needs --tasks")
        jobs = plan.build(cfg, Paths.of(cfg).config())
        names = task_list.numbered(cfg, jobs)
        providers = task_list.parse(args.tasks, names)
        reader = names[args.reader - 1] if args.reader else None
        if args.reader and not 1 <= args.reader <= len(names):
            parser.error(f"there is no task {args.reader}")
        out = args.out or task_list.default_archive(cfg, [names.index(t) + 1 for t in providers], args.reader)
        path = task_list.pack(cfg, providers, reader, out)
        print(f"[pack] {path} ({path.stat().st_size / 2**20:.1f} MB)")
    elif command == "unpack":
        if len(args.items) != 1:
            parser.error("unpack takes one archive")
        files = task_list.unpack(cfg, Path(args.items[0]))
        print(f"[unpack] {len(files)} files into {Paths.of(cfg).root}:\n  " + "\n  ".join(files[:20])
              + (f"\n  ... and {len(files) - 20} more" if len(files) > 20 else ""))
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
    elif command == "report":
        from vig_dagger import report

        report.write(cfg)
    elif command == "analyze":
        from vig_dagger.config import collections

        chosen = [Collection.parse(args.collection)] if args.collection else [
            c for c in collections(cfg) if (Paths.of(cfg).dagger(c) / "settings.json").exists()]
        for c in chosen:
            steps.analyze(cfg, c)
    elif command == "download-demos":
        hub.download_demos(cfg)
    elif command == "download-moe":
        if args.seed is None:
            parser.error("download-moe needs --seed")
        hub.download_moe(cfg, args.seed)
    elif command == "upload":
        for what in args.items or ["demos"]:
            hub.upload(cfg, what)
    elif command == "import-demos":
        hub.import_demos(cfg, [Path(p) for p in args.items])
    elif command == "import-moe":
        if args.seed is None or len(args.items) != 1:
            parser.error("import-moe takes one checkpoint and --seed")
        hub.import_moe(cfg, args.seed, Path(args.items[0]))
    else:
        parser.error(f"unknown command {command!r}; see vig-dagger --help")


if __name__ == "__main__":
    main()
