from pathlib import Path
import shutil
import subprocess
import sys


def run_command(args):
    subprocess.run(args, check=False)


def main():
    try:
        import questionary
    except ImportError:
        print("This menu requires 'questionary'. Install it with: pip install questionary")
        sys.exit(1)

    style = questionary.Style([
        ("qmark", "fg:#00d7ff bold"),
        ("question", "bold"),
        ("answer", "fg:#00d7ff bold"),
        ("pointer", "fg:#00d7ff bold"),
        ("highlighted", "fg:#ffffff bg:#005f87 bold"),
        ("selected", "fg:#00d7ff"),
        ("separator", "fg:#6c6c6c"),
        ("instruction", "fg:#6c6c6c"),
    ])

    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / "src"

    selection = questionary.select(
        "Select a step :",
        choices=[
            "Run all",
            "Distribution",
            "Augmentation",
            "Transformation",
            "Clean outputs only",
        ],
        style=style,
    ).ask()

    if not selection:
        print("No step selected.")
        return

    clean_outputs = False
    steps = []
    if selection == "Run all":
        steps = ["Distribution", "Augmentation", "Transformation"]
        clean_outputs = questionary.confirm(
            "Clean output directory first?",
            default=True,
            style=style,
        ).ask()
    elif selection == "Clean outputs only":
        clean_outputs = True
    else:
        steps = [selection]

    if selection == "Clean outputs only":
        output_dir = questionary.text(
            "Output directory to clean:",
            default="output/",
            style=style,
        ).ask()
        output_path = Path(output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        print(f"Cleaned: {output_path}")
        return

    dataset_path = None
    output_dir = None
    if "Distribution" in steps:
        dataset_path = questionary.text(
            "Dataset path:",
            default="dataset/images",
            style=style,
        ).ask()
        output_dir = questionary.text(
            "Output directory:",
            default="output/",
            style=style,
        ).ask()

        dataset_path = str(Path(dataset_path))
        output_dir = str(Path(output_dir))

        if clean_outputs:
            output_path = Path(output_dir)
            if output_path.exists():
                shutil.rmtree(output_path)
            print(f"Cleaned: {output_path}")

    if "Distribution" in steps and dataset_path and output_dir:
        run_command(
            [
                sys.executable,
                str(src_dir / "Distribution.py"),
                dataset_path,
                "--output-dir",
                output_dir,
            ]
        )

    if "Augmentation" in steps:
        aug_input = questionary.text(
            "Augmentation input (image or folder):",
            default=dataset_path or "dataset/images",
            style=style,
        ).ask()
        aug_output = questionary.text(
            "Augmentation output directory:",
            default=str(Path(output_dir or "output/") / "augmented"),
            style=style,
        ).ask()
        aug_input_path = Path(aug_input)

        command = [
            sys.executable,
            str(src_dir / "Augmentation.py"),
            str(aug_input_path),
            "--output-dir",
            str(Path(aug_output)),
        ]
        run_command(command)

    if "Transformation" in steps:
        default_src = (
            str(Path(output_dir) / "transformed_data")
            if output_dir
            else "dataset/images"
        )
        default_dst = (
            str(Path(output_dir) / "transformed_output")
            if output_dir
            else "output/transformed_output"
        )
        transform_src = questionary.text(
            "Transformation source directory:",
            default=default_src,
            style=style,
        ).ask()
        transform_dst = questionary.text(
            "Transformation destination directory:",
            default=default_dst,
            style=style,
        ).ask()

        run_command(
            [
                sys.executable,
                str(src_dir / "Transformation.py"),
                "-src",
                str(Path(transform_src)),
                "-dst",
                str(Path(transform_dst)),
            ]
        )


if __name__ == "__main__":
    main()
