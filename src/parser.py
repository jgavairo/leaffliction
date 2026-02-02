import argparse
from pathlib import Path


def parse_args():
    """
    Parse command-line arguments for the data analyzer tool.
    """
    parser = argparse.ArgumentParser(description="Data Analyzer Tool")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the input data directory."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the analysis results."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    return parser.parse_args()


def list_class_dirs(data_dir: Path):
    """
    List all class directories in the given root directory.

    Args:
        root_dir (Path): The root directory containing class subdirectories.

    Returns:
        List[Path]: A list of paths to class directories
    """
    root_dir = Path(data_dir)
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    return sorted(class_dirs, key=lambda d: d.name)


def count_img_files(class_dir: Path):
    """
    Count the number of image files in a given class directory.

    Args:
        class_dir (Path): The class directory to count image files in.

    Returns:
        int: The number of image files in the directory.
    """
    img_extensions = {".jpg", ".jpeg", ".png"}
    count = sum(
        1
        for file in class_dir.iterdir()
        if file.suffix.lower() in img_extensions
    )
    return count


def collect_distribution(data_list: list[Path]):
    """
    Collect the distribution of image files across class directories.

    Args:
        data_list (list[Path]): A list of class directories.

    Returns:
        tuple: A tuple containing a list of class directories and a dictionary mapping class names to image counts.
    """
    distribution = {}
    print("Class directories found:")
    for class_dir in data_list:
        nb = count_img_files(class_dir)
        distribution[class_dir.name] = nb
    return distribution
        
    