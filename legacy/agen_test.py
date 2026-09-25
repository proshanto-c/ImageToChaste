from import_modules import *

matplotlib.use("Agg")

checkpoint = "sam/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("Loading image...")
data_folder = "data"
image_folder = os.path.join(data_folder, "CLAHE_output")

output_folder = os.path.join("outs", "AGHPS")

# Load image in as RGB (y, x, c)
greyscale = True
image = Image.open(os.path.join(image_folder, "000zoomed.png"))
if greyscale:
    image = np.array(image.convert("RGB"))

filter_thresholds = [0.25, 0.5, 0.75]
stab_thresholds = [0, 0.25, 0.5, 0.75]
c_layer_depths = [0, 1, 2]
min_disc_sizes = [1, 10, 20]
m2ms = [False, True]

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

mask_generator = SAM2AutomaticMaskGenerator(model=build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False), 
                                            points_per_batch=128, pred_iou_thresh=0.75, stability_score_thresh=0.2, crop_n_layers=2, min_mask_region_area=10,
                                            use_m2m=False)



masks = mask_generator.generate(image)

print(len(masks))

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig(os.path.join(output_folder, f"zoom_f.75_s.2_c2_disc10_m2m0.png"))
plt.clf()
plt.close()

# for filter_thres in filter_thresholds:
#     for stab_thres in stab_thresholds:
#         for c_layers in c_layer_depths:
#             for disc_size in min_disc_sizes:
#                 for usem2m in m2ms:
#                     if usem2m == False:
#                         m = 0
#                     else:
#                         m = 1
                    
#                     mask_generator = SAM2AutomaticMaskGenerator(model=build_sam2(model_cfg, checkpoint, device=torch.device("cuda"), apply_postprocessing=False),
#                                                                 points_per_batch=128, pred_iou_thresh=filter_thres, stability_score_thresh=stab_thres, crop_n_layers=c_layers,
#                                                                 min_mask_region_area=disc_size, use_m2m=usem2m)

#                     masks = mask_generator.generate(image)

#                     print(len(masks))

#                     plt.figure(figsize=(20, 20))
#                     plt.imshow(image)
#                     show_anns(masks)
#                     plt.axis('off')
#                     plt.savefig(os.path.join(output_folder, f"f{filter_thres}_s{stab_thres}_c{c_layers}_disc{disc_size}_m2m{m}.png"))
#                     plt.clf()
#                     plt.close()