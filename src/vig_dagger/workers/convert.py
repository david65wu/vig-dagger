"""Convert Isaac Lab demo files into a training dataset (see ``vig_dagger/dataset.py``).

--sources is a JSON list of [demo .hdf5, demo indices or null for all]. The dataset is built in
``<out>.tmp`` and moved to --out when complete, so an interrupted job leaves no half dataset behind.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from vig_dagger import dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first-shard", type=int, default=0)
    args = p.parse_args()
    tmp = args.out.with_name(args.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    sources = [(Path(path), indices) for path, indices in json.loads(args.sources)]
    n = dataset.convert(sources, tmp, first_shard=args.first_shard)
    dataset.finalize(tmp, args.out)
    print(f"converted {n} demos -> {args.out}")


if __name__ == "__main__":
    main()
