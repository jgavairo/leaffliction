import argparse
from pathlib import Path

import parser
import plots


def parse_args():
    parser_args = argparse.ArgumentParser(
        description="Distribution analysis for image dataset"
    )
    parser_args.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset root (containing class subfolders)",
    )
    parser_args.add_argument(
        "--output-dir",
        type=str,
        default="distribution_graphs",
        help="Directory to save plots",
    )
    return parser_args.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    class_dirs = parser.list_class_dirs(data_dir)
    distribution = parser.collect_distribution(class_dirs)

    print("Classes found:")
    for class_name in distribution.keys():
        print(f"- {class_name}")

    print("\nImages per class:")
    for class_name, count in distribution.items():
        print(f"{class_name:<20}: {count:>5} image files")

    labels = list(distribution.keys())
    counts = list(distribution.values())

    plots.plot_bar(labels, counts, output_dir / "bar_chart.png")
    plots.plot_pie(labels, counts, output_dir / "pie_chart.png")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
