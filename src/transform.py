import cv2 as cv
import numpy as np
from pathlib import Path

def transform_images(image_path: Path, output_path: Path):
    """
    Apply a series of transformations to an image and save the result.

    Args:
        image_path (Path): Path to the input image.
        output_path (Path): Path to save the transformed image.
    """
    img_extensions = {".jpg", ".jpeg", ".png"}
    images = [f for f in image_path.iterdir() if f.suffix.lower() in img_extensions]
    if not images:
        return
    for img_file in images:
        image = cv.imread(str(img_file))
        if image is None:
            continue

        # Leaf segmentation using HSV mask (green range)
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower = np.array([25, 40, 40])
        upper = np.array([85, 255, 255])
        mask = cv.inRange(hsv, lower, upper)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        segmented = cv.bitwise_and(image, image, mask=mask)

        output_class_dir = output_path / image_path.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        output_img_path = output_class_dir / f"transformed_{img_file.name}"
        cv.imwrite(str(output_img_path), segmented)