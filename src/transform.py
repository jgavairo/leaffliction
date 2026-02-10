import cv2 as cv
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _process_image(img_path: Path, output_class_dir: Path):
    image = cv.imread(str(img_path))
    if image is None:
        return False

    # Leaf segmentation using GrabCut (faster settings)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    height, width = image.shape[:2]
    margin = 5
    rect = (margin, margin, width - 2 * margin, height - 2 * margin)
    cv.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        2,
        cv.GC_INIT_WITH_RECT,
    )

    fg_mask = (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD)
    mask = np.where(fg_mask, 255, 0).astype("uint8")
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    if mask.mean() < 5:
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        green_lower = np.array([20, 50, 40])
        green_upper = np.array([105, 255, 255])
        brown_lower = np.array([5, 50, 30])
        brown_upper = np.array([25, 255, 220])
        mask_green = cv.inRange(hsv, green_lower, green_upper)
        mask_brown = cv.inRange(hsv, brown_lower, brown_upper)
        mask = cv.bitwise_or(mask_green, mask_brown)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        segmented = cv.bitwise_and(image, image, mask=mask)
    else:
        segmented = cv.bitwise_and(image, image, mask=mask)

    segmented = cv.GaussianBlur(segmented, (5, 5), 0)
    output_img_path = output_class_dir / f"transformed_{img_path.name}"
    return cv.imwrite(str(output_img_path), segmented)


def transform_images(
    image_path: Path, output_path: Path, max_workers: int | None = None
):
    """
    Apply a series of transformations to an image and save the result.

    Args:
        image_path (Path): Path to the input image.
        output_path (Path): Path to save the transformed image.
    """
    img_extensions = {".jpg", ".jpeg", ".png"}
    images = [
        f for f in image_path.iterdir() if f.suffix.lower() in img_extensions
    ]
    if not images:
        return
    output_class_dir = output_path / image_path.name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_image, img_file, output_class_dir)
            for img_file in images
        ]
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Transforming {image_path.name}",
        ):
            pass
