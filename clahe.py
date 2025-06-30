import cv2
import os
import glob

inv = True

# Define paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_folder = os.path.join(project_dir, 'FF')
output_folder = os.path.join(project_dir, 'FF', 'CLAHE_INV_1')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Set up CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Apply CLAHE to each image in FF
for file_path in sorted(glob.glob(os.path.join(input_folder, '*.png'))):
    # Load the image in grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


    # Apply CLAHE
    clahe_img = clahe.apply(img)

    if inv==True:
        img = cv2.bitwise_not(img)
    # Apply Gaussian blur to create a "background" image
    background = cv2.GaussianBlur(clahe_img, (101, 101), 0)

    # Subtract the background from the original image
    flattened_img = cv2.subtract(clahe_img, background)

    # Rescale intensities to improve visibility
    # flattened_img = cv2.normalize(flattened_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Save the result
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_folder, base_name)
    cv2.imwrite(output_path, flattened_img)

    print(f"Processed {base_name} and saved to CLAHE_output folder.")
