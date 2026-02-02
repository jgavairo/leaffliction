import parser
from pathlib import Path
import plots


if __name__ == "__main__":
    """Main entry point for the data analyzer tool."""
    args = parser.parse_args()
    if args.verbose:
        print("Verbose mode is enabled.\n\n")
        print(f"Data Directory: {args.data_dir}")
        print(f"Output File: {args.output_file}\n\n")


    class_dirs = parser.list_class_dirs(Path(args.data_dir))
    distribution = parser.collect_distribution(class_dirs)

    ########################DEBUG OUTPUT########################
    if args.verbose:
        for class_dir, count in distribution.items():
            print(f"{class_dir:<20}: {count:>5} image files")
    ############################################################

    labels = list(distribution.keys())
    counts = list(distribution.values())

    plots.plot_bar(labels, counts, args.output_file + "bar_chart.png")

