import argparse
from pathlib import Path

import cv2
import shutil
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    RandomBrightnessContrast,
    GaussianBlur,
    GridDistortion,
)
import parser
import plots
from augment import augment_class


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data augmentation for images or folders"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image or a folder of images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/augmented",
        help="Directory to save augmented images",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=6,
        help="Number of augmented images per input image",
    )
    return parser.parse_args()


def build_augmentations():
    return Compose(
        [
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.2),
            Rotate(limit=20, p=0.5),
            RandomBrightnessContrast(p=0.5),
            GaussianBlur(p=0.3),
            GridDistortion(p=0.2),
        ]
    )


def is_image_file(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def iter_images(input_path: Path):
    if input_path.is_file() and is_image_file(input_path):
        return [input_path]
    return [p for p in input_path.rglob("*") if p.is_file() and is_image_file(p)]


def save_augmented(image_path: Path, output_root: Path, augmenter, variations: int):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for i in range(variations):
        augmented = augmenter(image=image)["image"]
        relative_parent = image_path.parent
        output_dir = output_root / relative_parent.name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = f"{image_path.stem}_aug_{i}{image_path.suffix}"
        cv2.imwrite(str(output_dir / output_name), augmented)


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        class_dirs = parser.list_class_dirs(input_path)
        distribution = parser.collect_distribution(class_dirs)
        if not distribution:
            print("No class folders found.")
            return

        max_count = max(distribution.values())
        missing_images = {
            class_name: max_count - count
            for class_name, count in distribution.items()
        }

        for class_name, missing in missing_images.items():
            if missing > 0:
                augment_class(input_path / class_name, missing, output_root)

        print(f"Augmented images saved to: {output_root}")
    else:
        augmenter = build_augmentations()
        images = iter_images(input_path)
        if not images:
            print("No images found.")
            return

        for img_path in images:
            save_augmented(img_path, output_root, augmenter, args.variations)

        print(f"Augmented images saved to: {output_root}")

    if input_path.is_dir():
        merged_dir = output_root / "merged"
        img_extensions = {".jpg", ".jpeg", ".png"}

        class_dirs = parser.list_class_dirs(input_path)
        for class_dir in class_dirs:
            source_dir = class_dir
            augmented_class_dir = output_root / class_dir.name
            merged_class_dir = merged_dir / class_dir.name
            merged_class_dir.mkdir(parents=True, exist_ok=True)

            if source_dir.exists():
                for img_path in source_dir.iterdir():
                    if img_path.suffix.lower() in img_extensions:
                        shutil.copy2(img_path, merged_class_dir / img_path.name)

            if augmented_class_dir.exists():
                for img_path in augmented_class_dir.iterdir():
                    if img_path.suffix.lower() in img_extensions:
                        shutil.copy2(img_path, merged_class_dir / img_path.name)

        distribution = parser.collect_distribution(parser.list_class_dirs(merged_dir))
        labels = list(distribution.keys())
        counts = list(distribution.values())
        plots.plot_bar(labels, counts, merged_dir / "bar_chart_merged.png")
        plots.plot_pie(labels, counts, merged_dir / "pie_chart_merged.png")
        print(f"Merged dataset saved to: {merged_dir}")


if __name__ == "__main__":
    main()
