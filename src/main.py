import parser
from pathlib import Path
import plots
from logger import logger


if __name__ == "__main__":
    """Main entry point for the data analyzer tool."""
    args = parser.parse_args()
    logger("Verbose mode is enabled.\n\n", args.verbose)
    logger(f"Data Directory: {args.data_dir}", args.verbose)
    logger(f"Output File: {args.output_file}\n\n", args.verbose)
    logger("Class directories found:", args.verbose)


    class_dirs = parser.list_class_dirs(Path(args.data_dir))
    distribution = parser.collect_distribution(class_dirs)

    for class_dir, count in distribution.items():
        logger(f"{class_dir:<20}: {count:>5} image files", args.verbose)
    logger("\n\n", args.verbose)

    labels = list(distribution.keys())
    counts = list(distribution.values())

    logger("Generating plots...", args.verbose)
    plots.plot_bar(labels, counts, Path(args.output_file) / "bar_chart.png")
    logger("Bar chart saved.", args.verbose)
    plots.plot_pie(labels, counts, Path(args.output_file) / "pie_chart.png")
    logger("Pie chart saved.", args.verbose)
    logger("All plots generated successfully.", args.verbose)

