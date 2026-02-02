import matplotlib.pyplot as plt


def plot_bar(labels, counts, output_path):
    """
    Plot a bar chart of the distribution of image files across class directories.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Class Directories')
    plt.ylabel('Number of Image Files')
    plt.title('Distribution of Image Files Across Class Directories')
    plt.savefig(output_path)
    plt.close()