import numpy as np

def mask_gen(amg, image):
    # Generate masks
    # Generate raw masks
    masks = amg.generate(image)

    # Stage 1: compute mean and std of mask areas
    areas = [np.sum(m['segmentation']) for m in masks]
    if len(areas) == 0:
        print("No masks found. Skipping filtering.")
        filtered_masks = []
    else:
        mean_area = np.mean(areas)
        std_area = np.std(areas)

        # Remove masks more than 3 std above the mean
        masks = [m for m, a in zip(masks, areas) if a <= mean_area + 3 * std_area]

        # Stage 2: recompute mean and std
        areas = [np.sum(m['segmentation']) for m in masks]
        mean_area = np.mean(areas)
        std_area = np.std(areas)

        # Remove masks more than 2 std above or below mean
        filtered_masks = [
            m for m, a in zip(masks, areas)
            if mean_area - 2 * std_area <= a <= mean_area + 2 * std_area
        ]

    num_masks = len(filtered_masks)
    print(f"Generated {num_masks} filtered masks.")

    return filtered_masks, num_masks