"""CLI entrypoint for disperse-wrapper.

Usage
-----
    disperse-wrapper run --points pts.npy --boxsize 100 --periodic --out outdir
    disperse-wrapper run --grid density.npy --boxsize 100 --out outdir
    disperse-wrapper convert --skel input.NDskl --to json --out out.json
    disperse-wrapper info --skel skeleton.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from .api import run_disperse_grid, run_disperse_points

    kwargs: dict = {}
    if args.binary_dir:
        kwargs["binary_dir"] = args.binary_dir

    out_dir = Path(args.out) if args.out else None

    if args.points:
        import numpy as np

        pts = np.load(args.points) if args.points.endswith(".npy") else args.points
        result = run_disperse_points(
            pts,
            args.boxsize,
            periodic=args.periodic,
            workdir=out_dir,
            persistence=args.persistence,
            smooth=args.smooth,
            keep_intermediate=args.keep_intermediate,
            **kwargs,
        )
    elif args.grid:
        result = run_disperse_grid(
            args.grid,
            args.boxsize,
            periodic=args.periodic,
            workdir=out_dir,
            persistence=args.persistence,
            smooth=args.smooth,
            keep_intermediate=args.keep_intermediate,
            **kwargs,
        )
    else:
        print("ERROR: provide --points or --grid", file=sys.stderr)
        return 2

    print(f"Run complete in {result.elapsed_seconds:.2f}s")
    print(f"  Work dir : {result.workdir}")
    print(f"  Files    : {[str(f) for f in result.skeleton_files]}")
    if result.skeleton:
        print(result.skeleton.summary())
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from .api import load_skeleton, save_skeleton

    sk = load_skeleton(args.skel)
    out_path = save_skeleton(sk, args.out, fmt=args.to)
    print(f"Saved to {out_path}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .api import load_skeleton

    sk = load_skeleton(args.skel)
    print(sk.summary())
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="disperse-wrapper",
        description="Python wrapper around DisPerSE command-line tools.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug output")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Run DisPerSE on point cloud or density grid")
    mode = run_p.add_mutually_exclusive_group()
    mode.add_argument("--points", metavar="POINTS_NPY", help="Path to (N,3) .npy point array")
    mode.add_argument("--grid", metavar="GRID_NPY", help="Path to (Nx,Ny,Nz) .npy density grid")
    run_p.add_argument("--boxsize", type=float, required=True, help="Box side length")
    run_p.add_argument("--periodic", action="store_true", default=False, help="Periodic BCs")
    run_p.add_argument("--out", metavar="OUTDIR", help="Output directory")
    run_p.add_argument("--persistence", type=float, default=None, help="Persistence threshold (sigma)")
    run_p.add_argument("--smooth", type=int, default=None, help="Smoothing iterations")
    run_p.add_argument("--binary-dir", default=None, help="Path to DisPerSE binaries directory")
    run_p.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")

    # --- convert ---
    conv_p = sub.add_parser("convert", help="Convert skeleton file format")
    conv_p.add_argument("--skel", required=True, metavar="INPUT", help="Input skeleton file")
    conv_p.add_argument("--to", required=True, choices=["json", "npz", "hdf5"], help="Output format")
    conv_p.add_argument("--out", required=True, metavar="OUTPUT", help="Output file path")

    # --- info ---
    info_p = sub.add_parser("info", help="Print skeleton statistics")
    info_p.add_argument("--skel", required=True, metavar="INPUT", help="Skeleton file to inspect")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "convert":
            return cmd_convert(args)
        if args.command == "info":
            return cmd_info(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
