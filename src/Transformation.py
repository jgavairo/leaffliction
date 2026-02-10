#!/usr/bin/env python3
"""Transformation.py - Image Transformation Tool

Applies 6 types of image transformations for leaf feature extraction.

Usage:
  ./Transformation.py <image_path>
  ./Transformation.py -src <dir> -dst <dir>

Transformations:
  Gaussian Blur, Mask, ROI Objects, Analyze Object,
  Pseudolandmarks, Color Histogram

Examples:
  ./Transformation.py ./Apple/image.JPG
  ./Transformation.py -src ./Apple/ -dst ./output/
"""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import csv
from skimage import measure, filters, morphology, color, exposure, util

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _print_progress(current: int, total: int, prefix: str):
    if total <= 0:
        return
    percent = int((current / total) * 100)
    bar_len = 24
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent}%)")
    sys.stdout.flush()


def is_image(filename):
    """Check if file is an image based on extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


class Transformation:
    """Image transformation class using scikit-image."""

    def __init__(self, image: np.ndarray):
        """Initialize with an image array."""
        self.img = image
        self.mask = self._create_mask()

    def _rgb(self):
        """Return RGB image as float in [0, 1] for scikit-image."""
        rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        return util.img_as_float32(rgb)

    def _create_mask(self):
        """Create binary mask to isolate leaf from background."""
        # Convert to HSV and extract saturation channel (skimage expects RGB)
        hsv = color.rgb2hsv(self._rgb())
        saturation = hsv[:, :, 1]

        # Apply binary threshold (fixed threshold similar to 58/255)
        binary = saturation > (58.0 / 255.0)
        return (binary.astype(np.uint8) * 255)

    def gaussian_blur(self):
        """Gaussian Blur transformation."""
        # Apply Gaussian blur to the mask for smoothing
        mask_bool = self.mask > 0
        blurred = filters.gaussian(
            mask_bool.astype(float), sigma=1.5, preserve_range=True
        )

        # Clean self.mask: fill holes and remove isolated pixels
        cleaned = morphology.binary_closing(mask_bool, morphology.disk(5))
        cleaned = morphology.binary_opening(cleaned, morphology.disk(7))
        self.mask = (cleaned.astype(np.uint8) * 255)

        # Return a 3-channel visualization of the blurred mask
        blurred_u8 = util.img_as_ubyte(np.clip(blurred, 0, 1))
        return cv.cvtColor(blurred_u8, cv.COLOR_GRAY2BGR)

    def masked_leaf(self):
        """Mask transformation - isolate leaf."""
        mask_bool = self.mask > 0
        result = self.img.copy()
        result[~mask_bool] = [255, 255, 255]

        # Remove dark pixels (0-50) from the mask for next transformations
        gray = color.rgb2gray(self._rgb())
        dark_pixels = gray <= (30.0 / 255.0)
        mask_bool[dark_pixels] = False

        # Fill the holes left by dark pixel removal
        mask_bool = morphology.binary_closing(mask_bool, morphology.disk(5))
        mask_bool = morphology.binary_opening(mask_bool, morphology.disk(5))
        self.mask = (mask_bool.astype(np.uint8) * 255)

        return result

    def roi_contours(self):
        """ROI Objects transformation - draw contours."""
        mask_bool = self.mask > 0
        result = self.img.copy()

        contours = measure.find_contours(mask_bool.astype(float), 0.5)
        if contours:
            # Fill the masked area in green
            result[mask_bool] = [0, 255, 0]

            labeled = measure.label(mask_bool)
            regions = measure.regionprops(labeled)
            if regions:
                largest = max(regions, key=lambda r: r.area)
                min_row, min_col, max_row, max_col = largest.bbox
                x, y, w, h = (
                    min_col,
                    min_row,
                    (max_col - min_col),
                    (max_row - min_row),
                )
                cv.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw contours in red
            for c in contours:
                pts = np.flip(c, axis=1).astype(np.int32)
                # (row, col) -> (x, y)
                if pts.shape[0] >= 2:
                    cv.polylines(
                        result,
                        [pts],
                        isClosed=True,
                        color=(0, 0, 255),
                        thickness=2,
                    )

        return result

    def analyze_shape(self):
        """Analyze shape and annotate basic metrics using scikit-image."""
        result = self.img.copy()
        mask_bool = self.mask > 0

        labeled = measure.label(mask_bool)
        regions = measure.regionprops(labeled)
        if not regions:
            return result

        largest = max(regions, key=lambda r: r.area)
        area = float(largest.area)
        perimeter = float(largest.perimeter)
        min_row, min_col, max_row, max_col = largest.bbox
        x, y, w, h = min_col, min_row, (max_col - min_col), (max_row - min_row)

        contours = measure.find_contours(mask_bool.astype(float), 0.5)
        for c in contours:
            pts = np.flip(c, axis=1).astype(np.int32)
            if pts.shape[0] >= 2:
                cv.polylines(
                    result,
                    [pts],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=2,
                )

        cv.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

        text = f"Area: {int(area)}  Perim: {int(perimeter)}"
        cv.putText(
            result,
            text,
            (x, max(10, y - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        return result

    def _draw_pseudolandmarks(self, img, pseudolandmarks, color, radius: int):
        """Draw pseudolandmark circles on image using OpenCV."""
        if pseudolandmarks is None or len(pseudolandmarks) == 0:
            return img

        for plm in pseudolandmarks:
            if plm is None or len(plm) == 0:
                continue

            pt = np.squeeze(np.asarray(plm))
            if pt.size < 2:
                continue

            x = int(pt[0])  # column
            y = int(pt[1])  # row

            # OpenCV expects (x, y) = (col, row)
            cv.circle(img, (x, y), radius, color, thickness=-1)

        return img

    def _pseudolandmarks_skimage(self):
        """Pseudolandmarks using scikit-image contours.

        Uses equal arc-length sampling.
        """
        result = self.img.copy()

        contours = measure.find_contours(self.mask, 0.5)
        if not contours:
            return result

        largest = max(contours, key=lambda c: c.shape[0])
        if largest.shape[0] < 2:
            return result

        # Sample N points along the contour by arc length
        n_points = 24
        diffs = np.diff(largest, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        cumdist = np.concatenate(([0.0], np.cumsum(dists)))
        total = float(cumdist[-1])
        if total <= 0:
            return result

        targets = np.linspace(0, total, n_points)
        landmarks = []
        for t in targets:
            idx = int(np.searchsorted(cumdist, t, side="left"))
            if idx <= 0:
                pt = largest[0]
            elif idx >= len(largest):
                pt = largest[-1]
            else:
                t0 = cumdist[idx - 1]
                t1 = cumdist[idx]
                if t1 > t0:
                    ratio = (t - t0) / (t1 - t0)
                else:
                    ratio = 0.0
                pt = largest[idx - 1] + ratio * (
                    largest[idx] - largest[idx - 1]
                )

            # skimage returns (row, col) -> OpenCV expects (x, y)
            x = int(pt[1])
            y = int(pt[0])
            landmarks.append((x, y))

        return self._draw_pseudolandmarks(
            result, landmarks, (0, 255, 0), radius=4
        )

    def pseudolandmarks(self):
        """Pseudolandmarks transformation."""
        return self._pseudolandmarks_skimage()

    def color_histogram(self):
        """Color Histogram transformation."""
        # Build histograms for RGB, HSV and LAB channels and return as image
        rgb = self._rgb()
        mask_bool = self.mask > 0

        channels = {
            'red': rgb[:, :, 0],
            'green': rgb[:, :, 1],
            'blue': rgb[:, :, 2],
        }

        hsv = color.rgb2hsv(rgb)
        channels['hue'] = hsv[:, :, 0]
        channels['saturation'] = hsv[:, :, 1]
        channels['value'] = hsv[:, :, 2]

        lab = color.rgb2lab(rgb)
        channels['lightness'] = lab[:, :, 0]
        channels['green-magenta'] = lab[:, :, 1]
        channels['blue-yellow'] = lab[:, :, 2]

        hist_data = {}
        for name, ch in channels.items():
            values = ch[mask_bool]
            if values.size == 0:
                hist = np.zeros(256, dtype=float)
            else:
                hist, _ = exposure.histogram(values, nbins=256)
                total = hist.sum() if hist.sum() != 0 else 1
                hist = (hist / total) * 100.0
            hist_data[name] = hist

        # Render a small figure to an image for saving
        fig, ax = plt.subplots(figsize=(3, 6), dpi=150)
        x = np.arange(256)
        color_map = {
            "blue": "b",
            "green": "g",
            "red": "r",
            "hue": "m",
            "saturation": "c",
            "value": "y",
            "lightness": "k",
            "green-magenta": "tab:olive",
            "blue-yellow": "tab:orange",
        }
        for name, hist in hist_data.items():
            ax.plot(
                x,
                hist,
                label=name,
                color=color_map.get(name, "k"),
                linewidth=1.0,
            )

        ax.set_xlabel("pixel intensity")
        ax.set_ylabel("proportion of pixels (%)")
        ax.legend(fontsize="xx-small")
        ax.set_xlim(0, 255)
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        hist_img = buf.reshape((height, width, 3))
        hist_img = cv.cvtColor(hist_img, cv.COLOR_RGB2BGR)
        plt.close(fig)
        return hist_img

    def color_histogram_data(self):
        """Return histogram data dict (name -> 256-array).

        Used for direct plotting.
        """
        rgb = self._rgb()
        mask_bool = self.mask > 0

        channels = {
            'red': rgb[:, :, 0],
            'green': rgb[:, :, 1],
            'blue': rgb[:, :, 2],
        }

        hsv = color.rgb2hsv(rgb)
        channels['hue'] = hsv[:, :, 0]
        channels['saturation'] = hsv[:, :, 1]
        channels['value'] = hsv[:, :, 2]

        lab = color.rgb2lab(rgb)
        channels['lightness'] = lab[:, :, 0]
        channels['green-magenta'] = lab[:, :, 1]
        channels['blue-yellow'] = lab[:, :, 2]

        hist_data = {}
        for name, ch in channels.items():
            values = ch[mask_bool]
            if values.size == 0:
                hist = np.zeros(256, dtype=float)
            else:
                hist, _ = exposure.histogram(values, nbins=256)
                total = hist.sum() if hist.sum() != 0 else 1
                hist = (hist / total) * 100.0
            hist_data[name] = hist

        return hist_data

    def get_all_transformations(self):
        """Get all transformation functions."""
        return {
            "GaussianBlur": self.gaussian_blur,
            "Mask": self.masked_leaf,
            "ROIObjects": self.roi_contours,
            "AnalyzeObject": self.analyze_shape,
            "Pseudolandmarks": self.pseudolandmarks,
            "ColorHistogram": self.color_histogram,
        }


def display_transformations(image_path, silent=False, max_display=200):
    """Display all transformations in a grid.

    silent: if True, suppress progress prints.
    """
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    transformer = Transformation(image)
    transforms = transformer.get_all_transformations()

    # Create smaller figure and compact GridSpec to reduce window size
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        3,
        hspace=0.12,
        wspace=0.12,
        left=0.02,
        right=0.98,
        top=0.9,
        bottom=0.03,
        width_ratios=[1.0, 1.0, 0.9],
        height_ratios=[1.0, 1.0, 1.0],
    )
    fig.suptitle(
        f"Image Transformations: {image_path.name}", fontsize=14
    )

    # Show original in top-left
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax_orig.set_title("Original")
    ax_orig.axis("off")

    # Define positions for transformations in 3-3-1 vertical layout
    # fig1(orig) fig4     ->  (0,0) (0,1)
    # fig2       fig5 fig7 ->  (1,0) (1,1) (1,2)
    # fig3       fig6     ->  (2,0) (2,1)
    # All except ColorHistogram
    transform_list = list(transforms.items())[:-1]
    positions = [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

    for idx, (name, func) in enumerate(transform_list):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])

        if not silent:
            print(f"  Applying {name}...")
        result = func()
        # Ensure single-channel images are converted to RGB for matplotlib
        if len(result.shape) == 2:
            display_img = cv.cvtColor(result, cv.COLOR_GRAY2RGB)
        else:
            display_img = cv.cvtColor(result, cv.COLOR_BGR2RGB)

        # Resize for display if larger than max_display (preserve aspect ratio)
        h, w = display_img.shape[:2]
        max_dim = max(h, w)
        if max_display and max_dim > max_display:
            scale = max_display / float(max_dim)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_img = cv.resize(
                display_img, (new_w, new_h), interpolation=cv.INTER_AREA
            )

        ax.imshow(display_img)
        ax.set_title(name)
        ax.axis("off")

    # ColorHistogram: plot directly on the axis for crisp vector rendering
    ax_hist = fig.add_subplot(gs[:, 2])
    if not silent:
        print("  Applying ColorHistogram...")
    try:
        hist_data = transformer.color_histogram_data()
        x = np.arange(256)
        color_map = {
            "blue": "b",
            "green": "g",
            "red": "r",
            "hue": "m",
            "saturation": "c",
            "value": "y",
            "lightness": "k",
            "green-magenta": "tab:olive",
            "blue-yellow": "tab:orange",
        }
        for name, hist in hist_data.items():
            ax_hist.plot(
                x,
                hist,
                label=name,
                color=color_map.get(name, "k"),
                linewidth=1.0,
            )
        ax_hist.set_xlabel("pixel intensity")
        ax_hist.set_ylabel("proportion of pixels (%)")
        ax_hist.set_xlim(0, 255)
        ax_hist.legend(fontsize="xx-small")
    except Exception:
        ax_hist.text(0.5, 0.5, "No histogram", ha="center")
    ax_hist.set_title("ColorHistogram")
    ax_hist.axis("on")

    # Show
    plt.show()


def save_transformations(
    image_path, output_dir, mask_only=False, silent=False
):
    """Save all transformations to files."""
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return []

    transformer = Transformation(image)
    base_name = image_path.stem
    extension = image_path.suffix

    output_paths = []

    if mask_only:
        # Only save mask
        try:
            result = transformer.masked_leaf()
            output_name = f"{base_name}_Mask{extension}"
            output_path = output_dir / output_name
            cv.imwrite(str(output_path), result)
            output_paths.append(output_path)
            if not silent:
                print(f"  Created: {output_name}")
        except Exception as e:
            print(f"  ✗ Mask failed: {e}")
    else:
        # Save all transformations
        transforms = transformer.get_all_transformations()
        for name, func in transforms.items():
            try:
                result = func()
                output_name = f"{base_name}_{name}{extension}"
                output_path = output_dir / output_name
                cv.imwrite(str(output_path), result)
                output_paths.append(output_path)
                if not silent:
                    print(f"  Created: {output_name}")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

    return output_paths


def save_dataset_entry(
    image_path: Path,
    output_root: Path,
    transformer: Transformation,
    target_size=(224, 224),
):
    """Save mask, resized normalized image and return features dict."""
    # Prepare output dirs
    rel_parent = image_path.parent.name
    masks_out = output_root / "masks" / rel_parent
    norm_out = output_root / "normalized" / rel_parent
    masks_out.mkdir(parents=True, exist_ok=True)
    norm_out.mkdir(parents=True, exist_ok=True)

    # Generate masked image and mask
    masked = transformer.masked_leaf()
    mask = transformer.mask

    # Compute shape features using largest contour
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    area = 0
    perimeter = 0
    bbox = (0, 0, 0, 0)
    if contours:
        largest = max(contours, key=cv.contourArea)
        area = float(cv.contourArea(largest))
        perimeter = float(cv.arcLength(largest, True))
        x, y, w, h = cv.boundingRect(largest)
        bbox = (int(x), int(y), int(w), int(h))

    # Mean color inside mask
    mean_bgr = cv.mean(transformer.img, mask=mask)[:3]
    mean_r, mean_g, mean_b = (
        float(mean_bgr[2]),
        float(mean_bgr[1]),
        float(mean_bgr[0]),
    )

    # Save files
    base_name = image_path.stem
    ext = image_path.suffix
    mask_name = f"{base_name}_mask.png"
    norm_name = f"{base_name}_norm{ext}"

    # ensure binary mask saved as single channel PNG
    cv.imwrite(str(masks_out / mask_name), mask)

    # Create normalized resized image (keep colors of masked)
    h, w = masked.shape[:2]
    norm = cv.resize(
        masked,
        (target_size[1], target_size[0]),
        interpolation=cv.INTER_AREA,
    )
    cv.imwrite(str(norm_out / norm_name), norm)

    # Feature dict
    features = {
        "file": str(image_path),
        "mask_path": str(masks_out / mask_name),
        "norm_path": str(norm_out / norm_name),
        "area": area,
        "perimeter": perimeter,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
    }

    return features


def process_directory(src_dir, dst_dir, mask_only=False, silent=False):
    """Process all images in source directory."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists() or not src_path.is_dir():
        print(f"Error: Invalid source directory: {src_dir}")
        sys.exit(1)

    dst_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = [
        f
        for f in src_path.rglob("*")
        if f.is_file() and is_image(f.name)
    ]

    if not images:
        print(f"No images found in: {src_dir}")
        sys.exit(1)

    mode = "mask transformations" if mask_only else "all transformations"
    print(f"\nProcessing {len(images)} images ({mode})...")

    # Prepare CSV for features when not in mask_only mode
    features_csv = dst_path / "features.csv"
    csv_fieldnames = [
        "file",
        "mask_path",
        "norm_path",
        "area",
        "perimeter",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "mean_r",
        "mean_g",
        "mean_b",
    ]

    if not mask_only:
        # Write header
        with open(features_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

    total = len(images)
    processed = 0

    for img_path in images:
        if mask_only:
            # Create subdirectory structure (preserve class folder)
            # only for mask mode
            rel_path = img_path.relative_to(src_path)
            output_subdir = dst_path / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            save_transformations(img_path, output_subdir, mask_only, silent)
            processed += 1
            if not silent:
                _print_progress(processed, total, "Masking")
            continue

        try:
            # For dataset mode: create transformer and save dataset entry
            image = cv.imread(str(img_path))
            if image is None:
                print(f"\n  ✗ Could not read: {img_path}")
                continue
            transformer = Transformation(image)
            features = save_dataset_entry(img_path, dst_path, transformer)

            # Append features to CSV
            with open(features_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerow(features)

            processed += 1
            if not silent:
                _print_progress(processed, total, "Transforming")
        except Exception as e:
            print(f"\n  ✗ Processing failed for {img_path.name}: {e}")

    if not silent:
        print()
    print(f"✓ Dataset transformations saved to: {dst_path}")


def main():
    """Main function."""
    try:
        # Silent mode
        silent = False
        if "-s" in sys.argv:
            silent = True
            sys.argv.remove("-s")

        # Show help
        if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
            print(__doc__)
            sys.exit(0)

        # Directory mode
        if "-src" in sys.argv:
            if "-dst" not in sys.argv:
                print("Error: -dst required when using -src")
                sys.exit(1)

            src_idx = sys.argv.index("-src")
            dst_idx = sys.argv.index("-dst")

            if src_idx + 1 >= len(sys.argv) or dst_idx + 1 >= len(sys.argv):
                print("Error: Missing directory path")
                sys.exit(1)

            src_dir = sys.argv[src_idx + 1]
            dst_dir = sys.argv[dst_idx + 1]
            mask_only = "-mask" in sys.argv

            process_directory(src_dir, dst_dir, mask_only, silent)
            return

        # Single image mode
        image_path = Path(sys.argv[1]).resolve()

        if not image_path.exists() or not image_path.is_file():
            print(f"Error: File not found: {image_path}")
            sys.exit(1)

        if not is_image(image_path.name):
            print(f"Error: Not a valid image: {image_path}")
            sys.exit(1)

        print(f"\nProcessing: {image_path.name}")
        print("Applying 6 transformations...\n")

        # Use a fixed, smaller max display dimension (no CLI option)
        max_display = 200
        display_transformations(image_path, silent, max_display)

    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
