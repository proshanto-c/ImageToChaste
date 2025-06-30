from import_modules import *
matplotlib.use("Agg")

checkpoints = "sam/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
greyscale = True

data_folder = "data"
image_folder = os.path.join(data_folder, "final_experiments")
output_folder = os.path.join("outs", "single_prep_2")
results_path = os.path.join("outs", "no_prep", "results_2.json")

images = ["009", "019", "039", "049", "059"]
actual_cell_counts = [310, 306, 276, 284, 297]

device = get_device()

amg = load_generator_from_run(8, results_path=results_path)
# model = build_sam2(model_cfg, checkpoints, device, apply_postprocessing=False)
# amg = SAM2AutomaticMaskGenerator(model, points_per_batch=128)

results = []

for idx, image_name in enumerate(images):
    print(f"Loading {image_name}")
    image = Image.open(os.path.join(image_folder, f"{image_name}.png"))
    if greyscale:
        image = np.array(image.convert("RGB"))

    filtered_masks, num_masks = mask_gen(amg, image)

    print(f"Generated {num_masks} masks for {image_name}.")

    short_filename = f"test2_masks_{image_name}.png"
    out_path = os.path.join(output_folder, short_filename)

    plt.figure()
    plt.imshow(image)
    show_anns(filtered_masks)
    plt.axis("off")
    plt.savefig(out_path)
    plt.clf()
    plt.close()

    perc_diff = round(100 * (num_masks - actual_cell_counts[idx]) / actual_cell_counts[idx], 2)

    results.append({
        "image": image_name,
        "mask_counts": num_masks,
        "percent_difference": perc_diff,
        "output_file": short_filename
    })

with open(os.path.join(output_folder, "test_results_2.json"), "w") as f:
    json.dump(results, f, indent=4)

