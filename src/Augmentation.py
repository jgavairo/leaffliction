import argparse
from pathlib import Path

import cv2
import shutil
import matplotlib.pyplot as plt
from albumentations import (
    Compose,
    Rotate,
    HueSaturationValue,
    RandomBrightnessContrast,
    RandomGamma,
    RandomScale,
    RandomResizedCrop,
    GaussianBlur,
    Perspective,
)
import parser
import plots
from augment import augment_class


def _print_progress(current: int, total: int, prefix: str):
    if total <= 0:
        return
    percent = int((current / total) * 100)
    bar_len = 24
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent}%)", end="", flush=True)


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
        default=None,
        help="Optional output directory override",
    )
    return parser.parse_args()


def build_augmentations():
    return Compose(
        [
            Rotate(limit=100, p=0.5),
            HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(40, 40), val_shift_limit=0, p=0.5),
            RandomBrightnessContrast(brightness_limit=(0.15, 0.15), contrast_limit=0, p=0.4),
            GaussianBlur(p=0.3),
            RandomScale(scale_limit=0.2, p=0.4),
            Perspective(scale=(0.08, 0.18), keep_size=True, fit_output=True, p=0.3),
        ]
    )


def build_demo_augmentations():
    return [
        ("rotation", Rotate(limit=100, p=1.0)),
        ("blur", GaussianBlur(blur_limit=50, p=1.0)),
        ("contrast", HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(60, 60), val_shift_limit=0, p=1.0)),
        ("scaling", RandomResizedCrop(size=(256, 256), scale=(0.5, 0.7), p=1.0)),
        ("illumination", RandomBrightnessContrast(brightness_limit=(0.4, 0.4), contrast_limit=0, p=1.0)),
        ("projective", Perspective(scale=(0.3, 1.0), keep_size=True, fit_output=True, p=1.0)),
    ]


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

    for i in range(6):  # Always use fixed 6 demo augmentations
        augmented = augmenter(image=image)["image"]
        relative_parent = image_path.parent
        output_dir = output_root / relative_parent.name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = f"{image_path.stem}_aug_{i}{image_path.suffix}"
        cv2.imwrite(str(output_dir / output_name), augmented)


def save_and_show_demo(image_path: Path, output_root: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    panels = [("original", image)]

    original_name = f"{image_path.stem}_original{image_path.suffix}"
    cv2.imwrite(str(output_root / original_name), image)

    for name, transform in build_demo_augmentations():
        augmented = transform(image=image)["image"]
        output_name = f"{image_path.stem}_{name}{image_path.suffix}"
        cv2.imwrite(str(output_root / output_name), augmented)
        panels.append((name, augmented))

    plt.figure(figsize=(12, 8))
    for idx, (title, img) in enumerate(panels, start=1):
        plt.subplot(2, 4, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if input_path.is_dir():
        output_root = Path(args.output_dir or "augmented_data")
        graphs_dir = Path("augmented_graphs")
    else:
        output_root = Path(args.output_dir or "augmented_data_demo")
        graphs_dir = None

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

        for class_name, count in distribution.items():
            source_dir = input_path / class_name
            output_class_dir = output_root / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            images = [p for p in source_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            total = len(images)
            copied = 0
            for img_path in images:
                shutil.copy2(img_path, output_class_dir / img_path.name)
                copied += 1
                _print_progress(copied, total, f"Copying {class_name}")
            if total > 0:
                print()

        for class_name, missing in missing_images.items():
            if missing > 0:
                augment_class(input_path / class_name, missing, output_root, verbose=True)

        print(f"Augmented images saved to: {output_root}")
    else:
        save_and_show_demo(input_path, output_root)
        print(f"Augmented images saved to: {output_root}")

    if input_path.is_dir():
        distribution = parser.collect_distribution(parser.list_class_dirs(output_root))
        labels = list(distribution.keys())
        counts = list(distribution.values())
        if graphs_dir is None:
            graphs_dir = Path("augmented_graphs")
        plots.plot_bar(labels, counts, graphs_dir / "bar_chart_merged.png")
        plots.plot_pie(labels, counts, graphs_dir / "pie_chart_merged.png")
        print("Plots saved for balanced dataset.")


if __name__ == "__main__":
    main()
