import numpy as np

def find_centroid(binary_image):
    # Get coordinates of all '1' pixels
    y_coords, x_coords = np.where(binary_image == 1)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None  # No shape found
    
    # Compute centroid
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    return centroid_x, centroid_y