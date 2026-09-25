from import_modules import *
from scipy.spatial import Voronoi, voronoi_plot_2d

amg = load_generator_from_run(4)

data_folder = "data"
image_folder = os.path.join(data_folder, "tune")
output_folder = os.path.join("outs", "amg_tuning_test")
os.makedirs(output_folder, exist_ok=True)

# Load image
print("Loading image...")
greyscale = True
image = Image.open(os.path.join(image_folder, "000_zoomed.png"))
if greyscale:
    image = np.array(image.convert("RGB"))

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

# --- Get centroids ---
centroids = []
for m in filtered_masks:
    centroid = find_centroid(m['segmentation'])
    if centroid is not None:
        centroids.append(centroid)

print(f"Computed {len(centroids)} centroids.")

# --- Voronoi Diagram ---
if len(centroids) >= 4:  # Voronoi requires at least 4 points to be interesting
    points = np.array(centroids)
    vor = Voronoi(points)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    # Overlay Voronoi
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='lime', line_width=1.2, point_size=2)
    ax.set_title("Voronoi Diagram of Mask Centroids")
    ax.set_axis_off()

    # Save
    voronoi_path = os.path.join(output_folder, "voronoi_overlay.png")
    plt.savefig(voronoi_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Voronoi diagram to: {voronoi_path}")

    # Convert to list of dicts for clarity
    centroids_list = [{"x": float(x), "y": float(y)} for x, y in centroids]

    # Write to JSON
    with open(os.path.join(output_folder, "cell_centroids.json"), "w") as f:
        json.dump(centroids_list, f, indent=2)
    
    with open(os.path.join(output_folder, "cell_centres.nodes"), "w") as f:
        f.write(f"{len(centroids)}\n")
        for idx, (x, y) in enumerate(centroids):
            f.write(f"{idx} {x:.6f} {y:.6f} 0\n")
    
else:
    print("Not enough centroids to generate Voronoi diagram.")
