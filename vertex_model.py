from import_modules import *
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage.morphology import skeletonize
import networkx as nx
matplotlib.use("Agg")

amg = load_generator_from_run(4)

data_folder = "data"
image_folder = os.path.join(data_folder, "tune")
output_folder = os.path.join("outs", "amg_tuning_test")
os.makedirs(output_folder, exist_ok=True)
mask_path = os.path.join(output_folder, 'masks.png')
skeleton_path = os.path.join(output_folder, 'skeleton_2.png')
out_path = os.path.join(output_folder, 'masks_2.png')
graph_path = os.path.join(output_folder, 'skeleton_graph_2.png')

# --- Load image ---
print("Loading image...")
greyscale = True
image = Image.open(os.path.join(image_folder, "059_zoomed.png"))
if greyscale:
    image = np.array(image.convert("RGB"))

# --- Generate masks ---
filtered_masks, num_masks = mask_gen(amg, image)

# --- Combine masks into a binary mask ---
print("Combining masks into a binary mask...")
combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for mask in filtered_masks:
    combined_mask = np.logical_or(combined_mask, mask['segmentation']).astype(np.uint8)
combined_mask *= 255

# Save and optionally display
print(f"Saving binary mask to {out_path}...")
Image.fromarray(combined_mask).save(out_path)

# --- Skeletonize the boundary ---
binary = (combined_mask < 128).astype(np.uint8)


skeleton = skeletonize(binary).astype(np.uint8)
# --- Add white border before skeletonization ---
print("Adding white border...")
padded_mask = np.pad(combined_mask, pad_width=1, mode='constant', constant_values=255)

# --- Detect junctions ---
kernel = np.ones((3, 3), dtype=np.uint8)
neighbor_count = cv2.filter2D(skeleton, -1, kernel) - skeleton
junction_mask = np.logical_and(skeleton == 1, neighbor_count >= 3)
junction_coords = set(tuple(coord) for coord in np.column_stack(np.where(junction_mask)))

# --- Build a graph from skeleton pixels ---
G = nx.Graph()
skeleton_coords = np.column_stack(np.where(skeleton > 0))

# Add edges between 8-connected neighbors
for y, x in skeleton_coords:
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                if skeleton[ny, nx_] == 1:
                    G.add_edge((y, x), (ny, nx_))

# --- Prune to junction-to-junction or endpoint-to-junction paths ---
# Define endpoints (==1 neighbor)
endpoints = [n for n in G.nodes if G.degree[n] == 1]
junctions = [n for n in G.nodes if G.degree[n] >= 3]

# Find all paths from junctions to junctions or endpoints
paths = []
visited = set()

for start in junctions + endpoints:
    if start in visited:
        continue
    for neighbor in G.neighbors(start):
        if (start, neighbor) in visited or (neighbor, start) in visited:
            continue
        path = [start, neighbor]
        visited.add((start, neighbor))
        curr = neighbor
        prev = start
        while curr not in junctions and G.degree[curr] == 2:
            next_ = [n for n in G.neighbors(curr) if n != prev][0]
            path.append(next_)
            visited.add((curr, next_))
            prev, curr = curr, next_
        if curr != path[-1]:
            path.append(curr)
        paths.append(path)

# --- Draw straight line segments between path endpoints ---
result = image.copy()
for path in paths:
    pt1 = (path[0][1] - 1, path[0][0] - 1)
    pt2 = (path[-1][1] - 1, path[-1][0] - 1)
    cv2.line(result, pt1, pt2, color=(0, 255, 0), thickness=1)

# --- Draw junctions ---
for y, x in junctions:
    cv2.circle(result, (x, y), radius=2, color=(255, 0, 0), thickness=-1)

# --- Save final output ---
print(f"Saving to {graph_path}...")
Image.fromarray(result).save(graph_path)



# from import_modules import *
# from scipy.spatial import Voronoi, voronoi_plot_2d
# from skimage.morphology import skeletonize
# matplotlib.use("Agg")

# amg = load_generator_from_run(4)

# data_folder = "data"
# image_folder = os.path.join(data_folder, "tune")
# output_folder = os.path.join("outs", "amg_tuning_test")
# os.makedirs(output_folder, exist_ok=True)
# out_path = os.path.join(output_folder, 'masks.png')

# # Load image
# print("Loading image...")
# greyscale = True
# image = Image.open(os.path.join(image_folder, "000_zoomed.png"))
# if greyscale:
#     image = np.array(image.convert("RGB"))

# # Generate masks
# filtered_masks, num_masks = mask_gen(amg, image)

# # Combine masks into a single binary mask
# print("Combining masks into a binary mask...")
# combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

# for mask in filtered_masks:
#     combined_mask = np.logical_or(combined_mask, mask['segmentation']).astype(np.uint8)

# # Scale mask to 0–255 (black and white)
# combined_mask *= 255

# # # Save and optionally display
# # print(f"Saving binary mask to {out_path}...")
# # Image.fromarray(combined_mask).save(out_path)

# # # Optional: view interactively
# # import matplotlib.pyplot as plt
# # plt.imshow(combined_mask, cmap='gray')
# # plt.axis('off')
# # plt.savefig(out_path)
# # plt.clf()
# # plt.close()

# binary = (combined_mask < 128).astype(np.uint8)  # black becomes 1 (foreground)
# skeleton = skeletonize(binary).astype(np.uint8)

# # Get coordinates of skeleton pixels
# coords = np.column_stack(np.where(skeleton > 0))

# # Define a 3x3 kernel for neighbor counting
# kernel = np.ones((3, 3), dtype=np.uint8)

# # Convolve to count neighbors (excluding center)
# neighbor_count = cv2.filter2D(skeleton, -1, kernel)
# neighbor_count -= skeleton  # remove self-count

# # Detect junctions (>=3 neighbors)
# junctions = np.logical_and(skeleton == 1, neighbor_count >= 3)
# junction_coords = np.column_stack(np.where(junctions))

# # Draw the skeleton and mark junctions
# out = np.stack([skeleton*255]*3, axis=2).astype(np.uint8)
# for y, x in junction_coords:
#     cv2.circle(out, (x, y), radius=2, color=(255, 0, 0), thickness=-1)

# # Show results
# out_path = os.path.join(output_folder, 'skeleton.png')
# plt.figure(figsize=(8, 8))
# plt.imshow(out)
# plt.title("Skeleton with Junctions")
# plt.axis("off")
# plt.savefig(out_path)
# plt.clf()
# plt.close()

# # segments = extract_expanded_edge_segments_from_disjoint_masks(filtered_masks, dilation_radius=8)
# # # Step 2: Clean them into a proper graph-like structure
# # connected_segments, endpoint_counts = clean_and_connect_segments(segments, rounding=1)

# # # print("Number of shared boundaries:", len(adjacency_list))
# # # print("First few edges (cell indices):", adjacency_list[:5])

# # plt.figure(figsize=(20, 20))
# # plt.imshow(image)
# # # for x0, y0, x1, y1 in connected_segments[:5]:
# # #     print(f"Shared edge between cell {i} and {j}: from ({x0}, {y0}) to ({x1}, {y1})")
# # for x0, y0, x1, y1 in connected_segments:
# #     plt.plot([x0, x1], [y0, y1], color='red', linewidth=1.5)
# # # Overlay the shared edge map in red
# # # plt.contour(shared_edge_map, levels=[0.5], colors='red', linewidths=1)

# # plt.axis("off")
# # plt.savefig(out_path)
# # plt.clf()
# # plt.close()