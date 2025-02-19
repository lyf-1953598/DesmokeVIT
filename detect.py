import cv2
import numpy as np
import os

def compute_color_statistics(image_path):
    """
    Compute mean Hue (HSV) and mean a* (CIELAB) values for a single image.

    Parameters:
    - image_path: str, path to the image file.

    Returns:
    - stats: dict, containing the computed mean Hue and mean a* values.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to HSV and CIELAB color spaces
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Extract and compute mean values for Hue and a* channels
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_a = np.mean(lab_image[:, :, 1])

    # Prepare results
    stats = {
        "mean_hue": mean_hue,
        "mean_a": mean_a,
    }
    return stats

def process_images_in_folder(folder_path):
    """
    Compute color statistics for all images in a given folder.

    Parameters:
    - folder_path: str, path to the folder containing image files.

    Returns:
    - stats_list: list of dict, containing mean Hue and mean a* values for each image.
    """
    stats_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            try:
                stats = compute_color_statistics(file_path)
                stats_list.append({"image_path": file_path, **stats})
            except ValueError as e:
                print(f"Error processing {file_path}: {e}")
    return stats_list

def compute_overall_statistics(stats_list):
    """
    Compute overall statistics (min, max, mean) for a list of image statistics.

    Parameters:
    - stats_list: list of dict, containing mean Hue and mean a* values for each image.

    Returns:
    - overall_stats: dict, containing overall min, max, and mean for Hue and a* values.
    """
    hues = [stats['mean_hue'] for stats in stats_list]
    a_values = [stats['mean_a'] for stats in stats_list]

    overall_stats = {
        "hue_min": np.min(hues),
        "hue_max": np.max(hues),
        "hue_mean": np.mean(hues),
        "a_min": np.min(a_values),
        "a_max": np.max(a_values),
        "a_mean": np.mean(a_values),
    }
    return overall_stats

# Example usage
if __name__ == "__main__":
    # Path to the folder containing images
    # folder_path = "./images"  # Replace with the actual folder path
    folder_path = "datasets/564"  # Replace with the actual folder path
    # folder_path = "datasets/hazy2clear_0206/trainA"  # Replace with the actual folder path
    # Process images in folder
    stats_list = process_images_in_folder(folder_path)

    # Compute overall statistics
    overall_stats = compute_overall_statistics(stats_list)

    # Display results
    print("Folder Image Color Statistics:")
    for stats in stats_list:
        print(f"Image: {stats['image_path']}, Mean Hue: {stats['mean_hue']:.2f}, Mean a*: {stats['mean_a']:.2f}")

    print("\nOverall Statistics:")
    print(f"Hue - Min: {overall_stats['hue_min']:.2f}, Max: {overall_stats['hue_max']:.2f}, Mean: {overall_stats['hue_mean']:.2f}")
    print(f"a*  - Min: {overall_stats['a_min']:.2f}, Max: {overall_stats['a_max']:.2f}, Mean: {overall_stats['a_mean']:.2f}")


# desmoke data gt：
# Overall Statistics:
# Hue - Min: 5.89, Max: 83.79, Mean: 22.89
# a*  - Min: 146.59, Max: 164.67, Mean: 156.44

# desmoke process data cyclegan：
# Overall Statistics:
# Hue - Min: 5.77, Max: 29.90, Mean: 12.83
# a*  - Min: 146.68, Max: 171.78, Mean: 157.86

# desmoke process data desmokevit：
# Overall Statistics:
# Hue - Min: 7.66, Max: 56.30, Mean: 24.08
# a*  - Min: 145.43, Max: 160.70, Mean: 153.28




# desmoke lap 0206 clear train:
# Overall Statistics:
# Hue - Min: 2.13, Max: 125.06, Mean: 15.38
# a*  - Min: 139.25, Max: 168.27, Mean: 155.19

# desmoke lap 0206 testb cyclegan :
# Overall Statistics:
# Hue - Min: 6.28, Max: 45.03, Mean: 17.50
# a*  - Min: 141.15, Max: 171.06, Mean: 155.75

# desmoke lap 0206 testb hazy2clear_0206_h2c :
# Overall Statistics:
# Hue - Min: 6.60, Max: 45.72, Mean: 18.30
# a*  - Min: 139.83, Max: 169.55, Mean: 155.56


# desmoke lap 0206 testb desmokevit :
# Overall Statistics:
# Hue - Min: 8.01, Max: 49.88, Mean: 22.24
# a*  - Min: 140.42, Max: 162.85, Mean: 153.68
