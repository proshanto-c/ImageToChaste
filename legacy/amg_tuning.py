from import_modules import *
matplotlib.use("Agg")

# Config
checkpoint = "sam/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

data_folder = "data"
image_folder = os.path.join(data_folder, "final_experiments")
output_folder = os.path.join("outs", "no_prep")
os.makedirs(output_folder, exist_ok=True)

# Load image
print("Loading image...")
greyscale = True
images = ['014.png', '029.png', '044.png']
actual_cell_counts = [306, 283, 276]

device = get_device()

# Parameter ranges (points_per_side is fixed)
param_space = {
    "points_per_batch": 128,
    "pred_iou_thresh": (0.0, 1.0),
    "stability_score_thresh": (0.0, 1.0),
    "crop_n_layers": [0, 1, 2],
    "crop_nms_thresh": (0.0, 1.0),
    "crop_overlap_ratio": (0.0, 0.5),
    "min_mask_region_area": (0, 20),
    "use_m2m": [False, True],
    "multimask_output": [True],
}

# User input
num_runs = int(input("Enter number of parameter search runs: "))

results = []

# Main loop
for run in range(num_runs):
    # Sample parameters
    params = {
        "points_per_side": 32,
        "points_per_batch": param_space["points_per_batch"],
        "pred_iou_thresh": round(random.uniform(*param_space["pred_iou_thresh"]), 2),
        "stability_score_thresh": round(random.uniform(*param_space["stability_score_thresh"]), 2),
        "crop_n_layers": random.choice(param_space["crop_n_layers"]),
        "crop_nms_thresh": round(random.uniform(*param_space["crop_nms_thresh"]), 2),
        "crop_overlap_ratio": round(random.uniform(*param_space["crop_overlap_ratio"]), 2),
        "min_mask_region_area": random.randint(*param_space["min_mask_region_area"]),
        "use_m2m": random.choice(param_space["use_m2m"]),
        "multimask_output": True,
    }

    print(f"\nRun {run+1}/{num_runs}: {params}")

    mask_counts = []
    perc_diffs = []
    output_files = []

    # Build mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        model=build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False),
        **params
    )

    for idx, image_file in enumerate(images):
        image = Image.open(os.path.join(image_folder, image_file))
        if greyscale:
            image = np.array(image.convert("RGB"))

        # Generate masks
        filtered_masks, num_masks = mask_gen(mask_generator, image)

        print(f"Generated {num_masks} masks for {image_file}.")

        # Save visual output
        short_filename = f"3rd_run_{run + 1}_{idx}.png"
        out_path = os.path.join(output_folder, short_filename)

        plt.figure()
        plt.imshow(image)
        show_anns(filtered_masks)
        plt.axis("off")
        plt.savefig(out_path)
        plt.clf()
        plt.close()

        mask_counts.append(num_masks)
        perc_diff = round(100 * (num_masks - actual_cell_counts[idx]) / actual_cell_counts[idx], 2)
        perc_diffs.append(perc_diff)
        output_files.append(short_filename)

    avg_perc_diff = round(sum(abs(p) for p in perc_diffs) / len(perc_diffs), 2)

    # Save run info
    results.append({
        "run": run + 1,
        "params": params,
        "mask_counts": mask_counts,
        "percent_differences": perc_diffs,
        "average_percent_difference": avg_perc_diff,
        "output_files": output_files
    })

# Write results to JSON
with open(os.path.join(output_folder, "results_3.json"), "w") as f:
    json.dump(results, f, indent=4)

# Print sorted comparison
print("\n--- Run comparison by average % difference from actual cell counts ---")
sorted_results = sorted(results, key=lambda x: x["average_percent_difference"])
for r in sorted_results:
    diffs_str = ", ".join([f"{n} ({'+' if d >= 0 else ''}{d}%)" for n, d in zip(r["mask_counts"], r["percent_differences"])])
    print(f"run {r['run']}: {diffs_str} | avg diff = {r['average_percent_difference']}%")

print(f"\nAll runs completed. Results saved to {os.path.join(output_folder, 'results.json')}")


# for idx, image_file in enumerate(images):
# idx = 2
# image_file = images[2]
# results = []
# data_folder = "data"
# image_folder = os.path.join(data_folder, "final_experiments")
# output_folder = os.path.join("outs", f"single_prep_{idx}")
# os.makedirs(output_folder, exist_ok=True)

# image = Image.open(os.path.join(image_folder, image_file))
# if greyscale:
#     image = np.array(image.convert("RGB"))

#     # Generate masks

# for run in range(num_runs):
#     # Sample parameters
#     params = {
#         "points_per_side": 32,
#         "points_per_batch": param_space["points_per_batch"],
#         "pred_iou_thresh": round(random.uniform(*param_space["pred_iou_thresh"]), 2),
#         "stability_score_thresh": round(random.uniform(*param_space["stability_score_thresh"]), 2),
#         "crop_n_layers": random.choice(param_space["crop_n_layers"]),
#         "crop_nms_thresh": round(random.uniform(*param_space["crop_nms_thresh"]), 2),
#         "crop_overlap_ratio": round(random.uniform(*param_space["crop_overlap_ratio"]), 2),
#         "min_mask_region_area": random.randint(*param_space["min_mask_region_area"]),
#         "use_m2m": random.choice(param_space["use_m2m"]),
#         "multimask_output": True,
#     }

#     print(f"\nRun {run+1}/{num_runs}: {params}")

#     # Build mask generator
#     mask_generator = SAM2AutomaticMaskGenerator(
#         model=build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False),
#         **params
#     )

#     filtered_masks, num_masks = mask_gen(mask_generator, image)

#     print(f"Generated {num_masks} masks for {image_file}.")

#     # Save visual output
#     short_filename = f"run_{run + 1}_{idx}.png"
#     out_path = os.path.join(output_folder, short_filename)

#     plt.figure()
#     plt.imshow(image)
#     show_anns(filtered_masks)
#     plt.axis("off")
#     plt.savefig(out_path)
#     plt.clf()
#     plt.close()

#     # mask_counts.append(num_masks)
#     perc_diff = round(100 * (num_masks - actual_cell_counts[idx]) / actual_cell_counts[idx], 2)
#     # perc_diffs.append(perc_diff)
#     # output_files.append(short_filename)

#     # avg_perc_diff = round(sum(abs(p) for p in perc_diffs) / len(perc_diffs), 2)

#     # Save run info
#     results.append({
#         "run": run + 1,
#         "params": params,
#         "mask_counts": num_masks,
#         "percent_differences": perc_diff,
#         "output_files": short_filename
#     })

# # Write results to JSON
# with open(os.path.join(output_folder, "results.json"), "w") as f:
#     json.dump(results, f, indent=4)