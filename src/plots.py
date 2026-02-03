import matplotlib.pyplot as plt
from pathlib import Path


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