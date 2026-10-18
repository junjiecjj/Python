"""Generate all nine example figures with one command."""

from importlib import import_module
from radar_utils import parse_args

PARTS = [
    "part1_signal_modeling",
    "part2_clutter_generation",
    "part3_pulse_compression",
    "part4_range_doppler",
    "part5_ca_cfar",
]


def main():
    args = parse_args(__doc__)
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    for name in PARTS:
        print(f"\n{'=' * 65}\n{name}\n{'=' * 65}")
        import_module(name).main(args.output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
