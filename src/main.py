import parser
from pathlib import Path
import shutil
import plots
from logger import Logger
from augment import augment_class
from transform import transform_images


if __name__ == "__main__":
    """Main entry point for the data analyzer tool."""

    args = parser.parse_args()
    logger = Logger(args.verbose).logger

    logger("Verbose mode is enabled.\n\n")
    logger(f"Data Directory: {args.data_dir}")
    logger(f"Output File: {args.output_file}\n\n")
    logger("Class directories found:")

    class_dirs = parser.list_class_dirs(Path(args.data_dir))
    distribution = parser.collect_distribution(class_dirs)

    for class_dir, count in distribution.items():
        logger(f"{class_dir:<20}: {count:>5} image files")
    logger("\n\n")

    labels = list(distribution.keys())
    counts = list(distribution.values())

    logger("Generating plots...")
    plots.plot_bar(labels, counts, Path(args.output_file) / "bar_chart.png")
    logger("Bar chart saved.")
    plots.plot_pie(labels, counts, Path(args.output_file) / "pie_chart.png")
    logger("Pie chart saved.")
    logger("All plots generated successfully.\n\n")

    logger("Calculation of missing images not implemented yet.")
    higher_count = max(counts) if counts else 0
    missing_images = {
        label: higher_count - count for label, count in distribution.items()
    }
    for class_dir, missing in missing_images.items():
        logger(f"{class_dir:<20}: {missing:>5} missing images")

    logger("\nData analysis completed.\n\n")

    logger("Starting data augmentation for classes with missing images...")
    output_dir = Path(args.output_file) / "augmented_data"
    for class_dir, missing in missing_images.items():
        if missing > 0:
            logger(f"Augmenting {missing} images for class {class_dir}")
            augment_class(Path(args.data_dir) / class_dir, missing, output_dir)
    logger("Data augmentation completed.\n\n")

    logger("Merging original and augmented datasets...")
    merged_dir = Path(args.output_file) / "merged"
    img_extensions = {".jpg", ".jpeg", ".png"}
    for class_name in distribution.keys():
        source_dir = Path(args.data_dir) / class_name
        augmented_class_dir = output_dir / class_name
        merged_class_dir = merged_dir / class_name
        merged_class_dir.mkdir(parents=True, exist_ok=True)

        if source_dir.exists():
            for img_path in source_dir.iterdir():
                if img_path.suffix.lower() in img_extensions:
                    shutil.copy2(img_path, merged_class_dir / img_path.name)

        if augmented_class_dir.exists():
            for img_path in augmented_class_dir.iterdir():
                if img_path.suffix.lower() in img_extensions:
                    shutil.copy2(img_path, merged_class_dir / img_path.name)

    logger("Merge completed. You can now run distribution on the merged folder.\n\n")

    logger("Create plots for the merged dataset...")
    merged_class_dirs = parser.list_class_dirs(merged_dir)
    merged_distribution = parser.collect_distribution(merged_class_dirs)
    merged_labels = list(merged_distribution.keys())
    merged_counts = list(merged_distribution.values())
    plots.plot_bar(merged_labels, merged_counts, merged_dir / "bar_chart_merged.png")
    logger("Bar chart for merged dataset saved.")
    plots.plot_pie(merged_labels, merged_counts, merged_dir / "pie_chart_merged.png")
    logger("Pie chart for merged dataset saved.")
    logger("All plots for merged dataset generated successfully.\n\n")

    logger("Transformation process started...")
    transformed_dir = Path(args.output_file) / "transformed_data"
    transformed_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in merged_class_dirs:
        transform_images(class_dir, transformed_dir)
    logger("Transformation process completed.\n\n")

    