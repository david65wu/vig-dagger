"""Command line: ``vig-dagger <command> -c <experiment.yaml> [--set key=value ...]``.

  check                  preflight: GPUs, the Isaac environment (starts Isaac Sim once), ffmpeg, disk, Hugging Face
                         and W&B access (run before `run`)
  run                    the whole experiment (every part of every round) as one queue of jobs; jobs start when
                         their inputs exist and the machine has room. Rerun after an interruption: finished jobs are
                         skipped, still-running ones adopted. --only REGEX restricts it to matching jobs (and what they
                         need); --dry-run lists the jobs.
  tasks                  the numbered tasks (several machines, assigned by hand): what each needs and who uses it
  run --tasks N ...      run only these tasks (numbers or ranges: 3 17 32-40) and the preparation they need; inputs
                         from other tasks must already be in this work directory
  pack --tasks N ... [--for M] [--out FILE]   one archive of these tasks' outputs (only what task M needs of them)
  unpack FILE            put an archive's files in place in this work directory
  status                 is the queue alive; failed and running jobs with their progress; milestones
  stop                   stop the queue and all its jobs (they resume where they stopped on the next `run`);
                         --queue-only: stop only the queue, the jobs keep running and the next `run` takes them over
  report                 write results/REPORT.md and results/figures/ from the results so far (any time; with
                         several machines, from every machine's results in the shared repo)
  release TASK           several machines: free a task whose machine is gone, so another machine takes it
  analyze --round R      recompute a round's DAgger metrics (results/, and W&B if enabled)
  summarize              recompute the evaluation summary (results/, and W&B if enabled)

Data and checkpoints:

  download-demos / download-policy   from the Hugging Face repos in hub.*
  upload {demos,policy,dagger}       to the Hugging Face repos in hub.*
  import-demos FILE.hdf5 ...         use existing planner demo files as the raw demos (part order = argument order;
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
    parser.add_argument("--tasks", nargs="+", help="run / pack: task numbers or ranges (see `vig-dagger tasks`)")
    parser.add_argument("--for", dest="for_task", help="pack: only what this task (a number) needs")
    parser.add_argument("--out", type=Path, help="pack: the archive to write (default: <work_dir>/packs/...)")
    args = parser.parse_intermixed_args()

    cfg = config_module.load(args.config, args.overrides)
    from vig_dagger.config import hub_mode
    from vig_dagger import hub, plan, steps

    command = args.command
    if command == "show-config":
        print(yaml.safe_dump(cfg, sort_keys=False))
    elif command == "run":
        from vig_dagger.config import Paths
        from vig_dagger.jobs import Queue, lock

        config_path = plan.write_config(cfg)
        jobs = plan.build(cfg, config_path, distributed=bool(args.tasks))
        selected = []
        if args.tasks:  # several machines, tasks assigned by hand: these tasks and what they need
            from vig_dagger import tasks as task_list

            names = task_list.numbered(cfg, jobs)
            selected = task_list.parse(args.tasks, names)
            problems = task_list.check_inputs(cfg, jobs, selected)
            if problems:
                raise SystemExit("cannot run yet; copy these outputs here first (`vig-dagger pack` on the machine that "
                                 "made them, `vig-dagger unpack` here):\n  " + "\n  ".join(problems))
            jobs = task_list.select_jobs(cfg, jobs, selected)
            print(f"[tasks] {', '.join(f'{names.index(t) + 1} ({t})' for t in selected)}", flush=True)
        if args.only:
            jobs = _select(jobs, args.only)
        if args.dry_run:
            for j in jobs:
                state = "done" if j.done is not None and j.done() else ""
                print(f"{j.priority:3d} {j.kind:5s} {j.name:60s} {state:5s} <- {', '.join(j.deps[:4])}{' ...' if len(j.deps) > 4 else ''}")
            return
        lock(Paths.of(cfg).root)
        scheduler = None
        if hub_mode(cfg):  # several machines: claim tasks through the shared repo
            from vig_dagger import board

            ok, who = board.token_ok()
            if not ok:
                raise SystemExit(f"Hugging Face: {who}")
            board.get(cfg).ensure_repo()
            print(f"[board] {cfg['sharing']['repo']} as {board.machine_id(cfg)} ({who})", flush=True)
            scheduler = lambda table: board.HubScheduler(cfg, table)  # noqa: E731
        done, stuck = Queue(jobs, cfg["resources"], Paths.of(cfg).logs(), hub=scheduler).run()
        if scheduler is not None:
            print(f"[queue] every task of the experiment is done ({len(done)} jobs here)")
            return
        print(f"[queue] finished: {len(done)} of {len(jobs)} jobs done" + (f"; failed or blocked: {sorted(stuck)}" if stuck else ""))
        if selected:
            from vig_dagger import tasks as task_list

            all_jobs = plan.build(cfg, config_path, distributed=True)
            rows = {r["task"]: r for r in task_list.table(cfg, all_jobs)}
            for t in selected:
                r = rows[t]
                state = "done" if task_list.present(cfg, t) else "NOT done"
                users = ", ".join(str(n) for n in r["used_by"]) or "none (the last task)"
                print(f"[tasks] task {r['number']} {state}; its outputs are used by tasks {users}"
                      + (f"; pass them on with: vig-dagger pack -c … --tasks {r['number']} --for <task>" if r["used_by"] else ""))
        if stuck:
            raise SystemExit(1)
    elif command == "tasks":
        from vig_dagger import tasks as task_list

        print(task_list.markdown(cfg, plan.build(cfg, plan.write_config(cfg), distributed=True)))
    elif command == "pack":
        from vig_dagger import tasks as task_list

        jobs = plan.build(cfg, plan.write_config(cfg), distributed=True)
        names = task_list.numbered(cfg, jobs)
        if not args.tasks:
            parser.error("pack needs --tasks (the tasks whose outputs to pack)")
        providers = task_list.parse(args.tasks, names)
        reader = task_list.parse([args.for_task], names)[0] if args.for_task else None
        out = args.out or task_list.default_archive(cfg, [names.index(t) + 1 for t in providers],
                                                   names.index(reader) + 1 if reader else None)
        path = task_list.pack(cfg, jobs, providers, reader, out)
        print(f"[pack] {path} ({path.stat().st_size / 2**20:.0f} MB); on the other machine: vig-dagger unpack -c … {path.name}")
    elif command == "unpack":
        from vig_dagger import tasks as task_list

        if len(args.items) != 1:
            parser.error("unpack takes one archive")
        files = task_list.unpack(cfg, Path(args.items[0]))
        print(f"[unpack] {len(files)} files into {cfg['work_dir']}")
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

        if hub_mode(cfg):  # every machine's results from the shared repo, summarized here
            steps.summarize(cfg)
        else:
            report.write(cfg)
    elif command == "release":
        from vig_dagger import board

        if len(args.items) != 1:
            parser.error("release takes one task, as vig-dagger status names it (e.g. train/retrain/base/seed1)")
        gone = board.get(cfg).release(args.items[0])
        print(f"released {args.items[0]}: removed {gone or 'nothing (it was not claimed)'}")
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
