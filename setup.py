#!/usr/bin/env python3
"""setup.py - Project CLI menu for Leaffliction pipeline.

Steps:
    1) Distribution analysis (plots)
    2) Split dataset/images -> dataset_train/ + dataset_predict/
    3) Augment dataset_train -> dataset_train_augmented/
    4) Transform train -> output_transform/train/
    5) Transform predict -> output_transform/predict/
    6) Full pipeline
    7) Demo augment random image
    8) Demo transform random image
    9) Clean generated outputs

You can run interactively or non-interactively:
  python setup.py
  python setup.py --list
  python setup.py --run 1
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
import subprocess
import curses

# Add src to import path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from Transformation import process_directory
except Exception:
    process_directory = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

DEFAULTS = {
    "source_dir": ROOT / "dataset" / "images",
    "train_dir": ROOT / "dataset_train",
    "predict_dir": ROOT / "dataset_predict",
    "aug_train_dir": ROOT / "dataset_train_augmented",
    "transform_out": ROOT / "output_transform",
}

# Paths created by the pipeline that can be cleaned
CLEAN_TARGETS = [
    DEFAULTS["train_dir"],
    DEFAULTS["predict_dir"],
    DEFAULTS["aug_train_dir"],
    DEFAULTS["transform_out"],
    ROOT / "augmented_graphs",
    ROOT / "distribution_graphs",
]


class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def fmt_path(path: Path) -> str:
    """Return a short, user-friendly path (relative to project root when possible)."""
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return path.name


def c(text: str, color: str) -> str:
    return f"{color}{text}{Style.RESET}"


def pick_random_image(root_dir: Path):
    """Pick a random image file from dataset/images (recursive)."""
    if not root_dir.exists():
        return None
    images = [p for p in root_dir.rglob("*") if p.is_file() and is_image_file(p)]
    if not images:
        return None
    return random.choice(images)


def print_step(title: str):
    line = "=" * (len(title) + 8)
    print(f"\n{c(line, Style.CYAN)}\n{c('=== ' + title + ' ===', Style.CYAN)}\n{c(line, Style.CYAN)}")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def split_dataset(source_dir: Path, train_dir: Path, predict_dir: Path, split_ratio=0.8, seed=42):
    """Split dataset by class folders into train and predict directories."""
    print_step("Step: Split dataset")
    if not source_dir.exists() or not source_dir.is_dir():
        print(c("✗ Source directory not found:", Style.RED), fmt_path(source_dir))
        return

    # Safety checks
    if train_dir.resolve() == source_dir.resolve() or predict_dir.resolve() == source_dir.resolve():
        print(c("✗ Train/Predict directories must be different from source directory.", Style.RED))
        return

    # Clear previous splits to avoid mixing old files
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if predict_dir.exists():
        shutil.rmtree(predict_dir)

    random.seed(seed)
    class_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        print("✗ No class directories found in source.")
        return

    for class_dir in class_dirs:
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image_file(p)]
        if not images:
            continue
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        predict_images = images[split_index:]

        train_class = train_dir / class_dir.name
        predict_class = predict_dir / class_dir.name
        train_class.mkdir(parents=True, exist_ok=True)
        predict_class.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_class / img.name)
        for img in predict_images:
            shutil.copy2(img, predict_class / img.name)

    print(c("✓ Split complete", Style.GREEN))
    print(f"  - Train: {fmt_path(train_dir)}")
    print(f"  - Predict: {fmt_path(predict_dir)}")


def augment_dataset(train_dir: Path, aug_out: Path):
    """Call Augmentation.py to balance and augment training dataset."""
    print_step("Step: Augment training dataset")
    if not train_dir.exists():
        print(c("✗ Train directory not found:", Style.RED), fmt_path(train_dir))
        return

    cmd = [sys.executable, str(SRC / "Augmentation.py"), str(train_dir), "--output-dir", str(aug_out)]
    print(c("→ Running augmentation", Style.YELLOW))
    print(f"  Script: Augmentation.py")
    print(f"  Input:  {fmt_path(train_dir)}")
    print(f"  Output: {fmt_path(aug_out)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(c("✓ Augmentation saved to:", Style.GREEN), fmt_path(aug_out))
    else:
        print(c("✗ Augmentation failed.", Style.RED))


def transform_dataset(source_dir: Path, output_dir: Path):
    """Run transformations and export dataset outputs using Transformation.py functions."""
    print_step("Step: Transform dataset")
    if process_directory is None:
        print(c("✗ Could not import Transformation.process_directory", Style.RED))
        return
    if not source_dir.exists():
        print(c("✗ Source directory not found:", Style.RED), fmt_path(source_dir))
        return

    print(c("→ Transforming:", Style.YELLOW), fmt_path(source_dir))
    print(c("→ Output:", Style.YELLOW), fmt_path(output_dir))

    process_directory(str(source_dir), str(output_dir), mask_only=False, silent=False)


def run_distribution(dataset_dir: Path):
    """Run Distribution.py to generate class distribution plots."""
    print_step("Step: Distribution analysis")
    if not dataset_dir.exists():
        print(c("✗ Dataset directory not found:", Style.RED), fmt_path(dataset_dir))
        return

    cmd = [sys.executable, str(SRC / "Distribution.py"), str(dataset_dir)]
    print(c("→ Running distribution analysis", Style.YELLOW))
    print(f"  Script: Distribution.py")
    print(f"  Input:  {fmt_path(dataset_dir)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(c("✓ Distribution plots generated.", Style.GREEN))
    else:
        print(c("✗ Distribution analysis failed.", Style.RED))


def demo_augment_random_image(source_dir: Path):
    """Run augmentation demo on a random image from dataset/images."""
    print_step("Demo: Augment random image")
    img = pick_random_image(source_dir)
    if img is None:
        print(c("✗ No image found in:", Style.RED), fmt_path(source_dir))
        return
    cmd = [sys.executable, str(SRC / "Augmentation.py"), str(img)]
    print(c("→ Running augmentation demo", Style.YELLOW))
    print(f"  Image: {fmt_path(img)}")
    subprocess.run(cmd)


def demo_transform_random_image(source_dir: Path):
    """Run transformation display on a random image from dataset/images."""
    print_step("Demo: Transform random image")
    img = pick_random_image(source_dir)
    if img is None:
        print(c("✗ No image found in:", Style.RED), fmt_path(source_dir))
        return
    cmd = [sys.executable, str(SRC / "Transformation.py"), str(img)]
    print(c("→ Running transform demo", Style.YELLOW))
    print(f"  Image: {fmt_path(img)}")
    subprocess.run(cmd)


def clean_pipeline_outputs():
    """Delete generated pipeline folders with confirmation."""
    print_step("Step: Clean outputs")
    existing = [p for p in CLEAN_TARGETS if p.exists()]
    if not existing:
        print(c("✓ Nothing to clean.", Style.GREEN))
        return

    print(c("The following folders will be removed:", Style.YELLOW))
    for p in existing:
        print(f"  - {fmt_path(p)}")

    confirm = input(c("Type 'YES' to confirm: ", Style.RED)).strip()
    if confirm != "YES":
        print(c("✗ Clean cancelled.", Style.RED))
        return

    for p in existing:
        shutil.rmtree(p, ignore_errors=True)

    print(c("✓ Clean complete.", Style.GREEN))


def menu():
    print("\nLeaffliction - Pipeline Menu")
    print("1) Distribution analysis (plots)")
    print("2) Split dataset (dataset/images -> dataset_train + dataset_predict)")
    print("3) Augment training dataset (dataset_train -> dataset_train_augmented)")
    print("4) Transform train dataset")
    print("5) Transform predict dataset")
    print("6) Run full pipeline (split -> augment -> transform train -> transform predict)")
    print("7) Demo augment random image (from dataset/images)")
    print("8) Demo transform random image (from dataset/images)")
    print("9) Clean generated outputs")
    print("0) Exit")


def run_choice(choice: str):
    src = DEFAULTS["source_dir"]
    train = DEFAULTS["train_dir"]
    predict = DEFAULTS["predict_dir"]
    aug = DEFAULTS["aug_train_dir"]
    out = DEFAULTS["transform_out"]

    if choice == "1":
        # Prefer augmented train if exists, else train, else source images
        if aug.exists():
            target = aug
        elif train.exists():
            target = train
        else:
            target = src
        run_distribution(target)
    elif choice == "2":
        split_dataset(src, train, predict)
    elif choice == "3":
        augment_dataset(train, aug)
    elif choice == "4":
        # Prefer augmented train if exists
        source = aug if aug.exists() else train
        transform_dataset(source, out / "train")
    elif choice == "5":
        transform_dataset(predict, out / "predict")
    elif choice == "6":
        split_dataset(src, train, predict)
        augment_dataset(train, aug)
        source = aug if aug.exists() else train
        transform_dataset(source, out / "train")
        transform_dataset(predict, out / "predict")
    elif choice == "7":
        demo_augment_random_image(src)
    elif choice == "8":
        demo_transform_random_image(src)
    elif choice == "9":
        clean_pipeline_outputs()
    elif choice == "0":
        print("Bye.")
    else:
        print("Invalid option.")


def curses_menu(options):
    """Interactive arrow-key menu using curses."""
    def _menu(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_WHITE, -1)

        current = 0
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            title = "Leaffliction Pipeline"
            subtitle = "Use ↑/↓ or j/k • Enter to select • Esc to quit"

            # Compute box size
            box_width = min(max(len(max([label for label, _ in options], key=len)) + 6, 40), width - 4)
            box_height = len(options) + 6

            start_y = max((height - box_height) // 2, 1)
            start_x = max((width - box_width) // 2, 2)

            # Draw border box
            for x in range(start_x, start_x + box_width):
                stdscr.addch(start_y, x, curses.ACS_HLINE)
                stdscr.addch(start_y + box_height - 1, x, curses.ACS_HLINE)
            for y in range(start_y, start_y + box_height):
                stdscr.addch(y, start_x, curses.ACS_VLINE)
                stdscr.addch(y, start_x + box_width - 1, curses.ACS_VLINE)
            stdscr.addch(start_y, start_x, curses.ACS_ULCORNER)
            stdscr.addch(start_y, start_x + box_width - 1, curses.ACS_URCORNER)
            stdscr.addch(start_y + box_height - 1, start_x, curses.ACS_LLCORNER)
            stdscr.addch(start_y + box_height - 1, start_x + box_width - 1, curses.ACS_LRCORNER)

            # Title
            title_x = start_x + (box_width - len(title)) // 2
            stdscr.addstr(start_y + 1, title_x, title, curses.color_pair(1) | curses.A_BOLD)

            # Options
            for idx, (label, _value) in enumerate(options):
                line_y = start_y + 3 + idx
                line_x = start_x + 2
                text = f"{idx + 1}. {label}"
                if idx == current:
                    stdscr.addstr(line_y, line_x, text.ljust(box_width - 4), curses.color_pair(2) | curses.A_BOLD)
                else:
                    stdscr.addstr(line_y, line_x, text.ljust(box_width - 4), curses.color_pair(4))

            # Footer help
            footer_x = start_x + (box_width - len(subtitle)) // 2
            stdscr.addstr(start_y + box_height - 2, footer_x, subtitle, curses.color_pair(3))

            stdscr.refresh()

            key = stdscr.getch()
            if key in (3, 27):  # Ctrl+C or ESC
                return "0"
            if key in (curses.KEY_UP, ord('k')):
                current = (current - 1) % len(options)
            elif key in (curses.KEY_DOWN, ord('j')):
                current = (current + 1) % len(options)
            elif key in (curses.KEY_ENTER, 10, 13):
                return options[current][1]

    return curses.wrapper(_menu)


def main():
    parser = argparse.ArgumentParser(description="Leaffliction pipeline CLI menu")
    parser.add_argument("--list", action="store_true", help="List menu options and exit")
    parser.add_argument("--run", type=str, default=None, help="Run a menu option directly (e.g., 1..9)")
    parser.add_argument("--text", action="store_true", help="Force text menu (no arrows)")
    args = parser.parse_args()

    if args.list:
        menu()
        return

    if args.run is not None:
        run_choice(args.run)
        return

    options = [
        ("Distribution analysis (plots)", "1"),
        ("Split dataset (dataset/images -> dataset_train + dataset_predict)", "2"),
        ("Augment training dataset (dataset_train -> dataset_train_augmented)", "3"),
        ("Transform train dataset", "4"),
        ("Transform predict dataset", "5"),
        ("Run full pipeline (split -> augment -> transform train -> transform predict)", "6"),
        ("Demo augment random image (from dataset/images)", "7"),
        ("Demo transform random image (from dataset/images)", "8"),
        ("Clean generated outputs", "9"),
        ("Exit", "0"),
    ]

    if not args.text:
        try:
            while True:
                choice = curses_menu(options)
                if choice == "0":
                    run_choice(choice)
                    break
                run_choice(choice)
                input("\nPress Enter to return to the menu...")
            return
        except KeyboardInterrupt:
            print("\nBye.")
            return
        except Exception:
            print("\n⚠ Arrow-key menu unavailable. Falling back to text menu.")

    # Fallback to simple text menu
    try:
        while True:
            menu()
            choice = input("Select an option: ").strip()
            if choice == "0":
                run_choice(choice)
                break
            run_choice(choice)
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
