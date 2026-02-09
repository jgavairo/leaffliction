import cv2
import numpy as np
from pathlib import Path
import sys

from albumentations import (
    Rotate,
    RandomResizedCrop,
    RandomBrightnessContrast,
    RandomGamma,
    RandomScale,
    GridDistortion,
    GaussianBlur,
    Perspective,
    Compose,
)

def augment_class(class_dir: Path, deficit: int, output_dir: Path, verbose: bool = True):
    """
    Augment images in the specified class directory to address the deficit.

    Args:
        class_dir (Path): The class directory to augment images in.
        deficit (int): The number of images to generate.
        output_dir (Path): The directory to save augmented images.

    Returns:
        None
    """
    img_extensions = {".jpg", ".jpeg", ".png"}
    images = [f for f in class_dir.iterdir() if f.suffix.lower() in img_extensions]
    if not images:
        return

    augmentations = Compose([
        Rotate(limit=20, p=0.5),
        RandomResizedCrop(size=(224, 224), p=0.4),
        RandomScale(scale_limit=0.2, p=0.4),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.4),
        GaussianBlur(p=0.3),
        Perspective(scale=(0.02, 0.08), p=0.3),
    ])

    output_class_dir = output_dir / class_dir.name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    attempts = 0
    max_attempts = max(deficit * 5, 10)
    while generated < deficit and attempts < max_attempts:
        attempts += 1
        img_path = np.random.choice(images)
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        augmented = augmentations(image=image)
        augmented_image = augmented["image"]

        output_path = output_class_dir / f"augmented_{generated}_{img_path.name}"
        if cv2.imwrite(str(output_path), augmented_image):
            generated += 1
            if verbose:
                percent = int((generated / deficit) * 100) if deficit > 0 else 100
                bar_len = 20
                filled = int(bar_len * generated / max(deficit, 1))
                bar = "#" * filled + "-" * (bar_len - filled)
                sys.stdout.write(f"\r    Augmenting {class_dir.name}: [{bar}] {generated}/{deficit} ({percent}%)")
                sys.stdout.flush()

    if verbose and deficit > 0:
        print()
    