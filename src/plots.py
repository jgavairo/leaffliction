import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np


def plot_bar(labels, counts, output_path):
    """
    Plot a bar chart of the distribution of image files across class directories.
    """
    if not labels or not counts:
        raise ValueError("Labels and counts must not be empty.")
    plt.figure(figsize=(16, 9))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Class Directories')
    plt.ylabel('Number of Image Files')
    plt.title('Distribution of Image Files Across Class Directories')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_pie(labels, counts, output_path):
    """
    Plot a pie chart of the distribution of image files across class directories.
    """
    if not labels or not counts:
        raise ValueError("Labels and counts must not be empty.")
    plt.figure(figsize=(16, 9))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Image Files Across Class Directories')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_histogram(data, output_path, bins=256):
    """
    Plot histogram(s).

    - If data is a sequence of values, plot a single histogram.
    - If data is a folder path, generate per-class RGB and HSV histograms.
    """
    if data is None:
        raise ValueError("Data must not be empty.")

    if isinstance(data, (list, tuple, np.ndarray)):
        if len(data) == 0:
            raise ValueError("Data must not be empty.")
        plt.figure(figsize=(16, 9))
        plt.hist(data, bins=bins, color="skyblue", edgecolor="black")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        return

    input_dir = Path(data)
    if not input_dir.exists():
        raise ValueError("Input directory does not exist.")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png"}
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    for class_dir in sorted(class_dirs, key=lambda d: d.name):
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in img_extensions
        ]
        if not images:
            continue

        rgb_hist = np.zeros((3, bins), dtype=np.float64)
        hsv_hist = np.zeros((3, bins), dtype=np.float64)
        count = 0

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            gray_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)[1]

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            for channel in range(3):
                rgb_hist[channel] += cv2.calcHist(
                    [rgb], [channel], mask, [bins], [0, 256]
                ).flatten()
                hsv_hist[channel] += cv2.calcHist(
                    [hsv], [channel], mask, [bins], [0, 256]
                ).flatten()

            count += 1

        if count == 0:
            continue

        rgb_hist /= count
        hsv_hist /= count

        class_output_dir = output_dir / class_dir.name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(16, 9))
        colors = ["r", "g", "b"]
        labels = ["R", "G", "B"]
        for idx, color in enumerate(colors):
            plt.plot(rgb_hist[idx], color=color, label=labels[idx])
        plt.title(f"RGB Histogram - {class_dir.name}")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(class_output_dir / "hist_rgb.png")
        plt.close()

        plt.figure(figsize=(16, 9))
        colors = ["m", "c", "y"]
        labels = ["H", "S", "V"]
        for idx, color in enumerate(colors):
            plt.plot(hsv_hist[idx], color=color, label=labels[idx])
        plt.title(f"HSV Histogram - {class_dir.name}")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(class_output_dir / "hist_hsv.png")
        plt.close()


if __name__ == "__main__":
    plot_histogram("output/transformed_data", "output/histograms")