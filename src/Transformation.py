import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image transformations and feature extraction"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=str,
        help="Single image path to display transformations",
    )
    parser.add_argument(
        "-src",
        dest="src",
        type=str,
        help="Source directory of images",
    )
    parser.add_argument(
        "-dst",
        dest="dst",
        type=str,
        help="Destination directory to save transformations (optional)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display transformations on screen",
    )
    return parser.parse_args()


def is_image_file(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def grabcut_mask(image: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    height, width = image.shape[:2]
    margin = 5
    rect = (margin, margin, width - 2 * margin, height - 2 * margin)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)

    mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if mask.mean() < 5:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_lower = np.array([20, 50, 40])
        green_upper = np.array([105, 255, 255])
        brown_lower = np.array([5, 50, 30])
        brown_upper = np.array([25, 255, 220])
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        mask_brown = cv2.inRange(hsv, brown_lower, brown_upper)
        mask = cv2.bitwise_or(mask_green, mask_brown)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)


def roi_objects(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    if not contours:
        return output
    largest = max(contours, key=cv2.contourArea)
    overlay = output.copy()
    cv2.drawContours(overlay, [largest], -1, (0, 255, 0), thickness=-1)
    output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return output


def object_analysis(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, [largest], -1, (0, 255, 0), 2)
        moments = cv2.moments(largest)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(output, (cx, cy), 4, (255, 0, 255), -1)
    return output


def pseudo_landmarks(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    if not contours:
        return output
    largest = max(contours, key=cv2.contourArea)
    step = max(len(largest) // 20, 1)
    for i in range(0, len(largest), step):
        point = tuple(largest[i][0])
        cv2.circle(output, point, 3, (0, 0, 255), -1)
    return output


def compute_color_hists(image: np.ndarray, mask: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    hist_r = cv2.calcHist([rgb], [0], mask, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([rgb], [1], mask, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([rgb], [2], mask, [256], [0, 256]).flatten()

    hist_h = cv2.calcHist([hsv], [0], mask, [256], [0, 256]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], mask, [256], [0, 256]).flatten()

    hist_l = cv2.calcHist([lab], [0], mask, [256], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab], [1], mask, [256], [0, 256]).flatten()
    hist_b_lab = cv2.calcHist([lab], [2], mask, [256], [0, 256]).flatten()

    hists = {
        "red": hist_r,
        "green": hist_g,
        "blue": hist_b,
        "hue": hist_h,
        "saturation": hist_s,
        "value": hist_v,
        "lightness": hist_l,
        "green-magenta": hist_a,
        "blue-yellow": hist_b_lab,
    }

    for key, hist in hists.items():
        total = hist.sum()
        if total > 0:
            hists[key] = (hist / total) * 100.0

    return hists


def color_histogram(image: np.ndarray, mask: np.ndarray, output_path: Path):
    hists = compute_color_hists(image, mask)
    plt.figure(figsize=(12, 6))
    color_map = {
        "red": "r",
        "green": "g",
        "blue": "b",
        "hue": "m",
        "saturation": "c",
        "value": "y",
        "lightness": "#666666",
        "green-magenta": "#ff66cc",
        "blue-yellow": "#ffee00",
    }
    for label, hist in hists.items():
        plt.plot(hist, color=color_map.get(label, "k"), label=label)
    plt.title("Color Histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def process_image(image_path: Path, output_dir: Path | None, show: bool):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    mask = grabcut_mask(image)
    masked = apply_mask(image, mask)
    blurred = gaussian_blur(image)
    roi = roi_objects(image, mask)
    analysis = object_analysis(masked, mask)
    landmarks = pseudo_landmarks(masked, mask)
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "original.png"), image)
        cv2.imwrite(str(output_dir / "gaussian_blur.png"), blurred)
        cv2.imwrite(str(output_dir / "mask.png"), mask_vis)
        cv2.imwrite(str(output_dir / "roi_objects.png"), roi)
        cv2.imwrite(str(output_dir / "object_analysis.png"), analysis)
        cv2.imwrite(str(output_dir / "pseudolandmarks.png"), landmarks)
        color_histogram(image, mask, output_dir / "hist_rgb.png")

    if show:
        panels = [
            ("Original", image),
            ("Gaussian Blur", blurred),
            ("Mask", mask_vis),
            ("ROI Objects", roi),
            ("Object Analysis", analysis),
            ("Pseudo-landmarks", landmarks),
        ]
        plt.figure(figsize=(14, 8))
        grid = plt.GridSpec(2, 4)
        for idx, (title, img) in enumerate(panels, start=1):
            plt.subplot(grid[(idx - 1) // 4, (idx - 1) % 4])
            if img.ndim == 2:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis("off")

        hists = compute_color_hists(image, mask)
        plt.subplot(grid[1, 2:4])
        color_map = {
            "red": "r",
            "green": "g",
            "blue": "b",
            "hue": "m",
            "saturation": "c",
            "value": "y",
            "lightness": "#666666",
            "green-magenta": "#ff66cc",
            "blue-yellow": "#ffee00",
        }
        for label, hist in hists.items():
            plt.plot(hist, color=color_map.get(label, "k"), label=label)
        plt.title("Color Histogram")
        plt.xlabel("Pixel intensity")
        plt.ylabel("Proportion of pixels (%)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.show()


def iter_images(input_path: Path):
    if input_path.is_file() and is_image_file(input_path):
        return [input_path]
    return [p for p in input_path.rglob("*") if p.is_file() and is_image_file(p)]


def main():
    args = parse_args()

    if args.src:
        src = Path(args.src)
        dst = Path(args.dst) if args.dst else Path("transformed_data")
        images = iter_images(src)
        if not images:
            print("No images found.")
            return
        for img_path in images:
            rel_parent = img_path.parent.name
            output_dir = dst / rel_parent / img_path.stem
            process_image(img_path, output_dir, show=False)
        print(f"Transformations saved to: {dst}")
        return

    if args.image_path:
        demo_dir = Path("transformed_data_demo")
        demo_dir.mkdir(parents=True, exist_ok=True)
        process_image(Path(args.image_path), output_dir=demo_dir, show=True)
        return

    print("Usage: provide an image path or use -src and -dst.")


if __name__ == "__main__":
    main()
