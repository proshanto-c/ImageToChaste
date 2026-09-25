from import_modules import *

matplotlib.use("Agg")

checkpoint = "sam/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

print("Loading image...")
data_folder = "data"
image_folder = os.path.join(data_folder, "CLAHE_output")

output_folder = os.path.join("outs", "2s_test_1")

# Load image in as RGB (y, x, c)
greyscale = True
image = Image.open(os.path.join(image_folder, "000zoomed.png"))
if greyscale:
    image = np.array(image.convert("RGB"))

print(image.dtype)

# Set target for first step segmentation
y, x, _ = image.shape
x2 = round(x/2)
y2 = round(y/2)
input_point1 = np.array([[x2, y2]])
input_label1 = np.array([1])

# Set target for second step segmentation
input_point2 = np.array([[20, 20]])
input_label2 = np.array([1])

# First prediction generation and print
print("Generating first prediction...")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # Place image into the predictor
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(point_coords=input_point1, point_labels=input_label1, multimask_output=True)

expanded_masks = []
masked_images = []
for i, (mask, score) in enumerate(zip(masks, scores)):
    # Expand the mask to being of shape (y, x, c) from (y, x)
    e_mask = np.expand_dims(mask.astype(np.uint8), axis=-1)
    e_mask = np.repeat(e_mask, 3, axis=-1)
    expanded_masks.append(e_mask)

    # Apply mask to the image
    m_image = image * e_mask
    masked_images.append(m_image)

    # Plot the masked image
    plt.figure(figsize=(10, 10))
    plt.imshow(m_image)
    # Place the mask on the image
    show_mask(mask, plt.gca())
    # Place the token on the image
    show_points(input_point1, input_label1, plt.gca())
    plt.title(f'Step 1 Mask {i+1}, Score: {score:.3f}', fontsize=18)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"zstep1_{i}.png"))
    plt.clf()
    plt.imshow(m_image)
    plt.title(f'Step 1 Mask {i+1}, Score: {score:.3f}', fontsize=18)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"zstep1_nm_{i}.png"))


desired_mask = int(input("Which mask is best (give the number as found on the image)?"))

mask_number = desired_mask - 1

# Isolate the selected mask to perform second step of segmentation
f_image = masked_images[mask_number]
print(f_image.dtype)
print(f_image.shape)


# Generate second prediction
print("Generating second prediction...")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # Place image into the predictor
    predictor.set_image(f_image)
    masks, scores, logits = predictor.predict(point_coords=input_point2, point_labels=input_label2, multimask_output=True)

expanded_masks2 = []
edges = []
for i, (mask, score) in enumerate(zip(masks, scores)):
    # Expand the mask to being of shape (y, x, c) from (y, x)
    e_mask = np.expand_dims(mask.astype("i"), axis=-1)
    e_mask = np.repeat(e_mask, 3, axis=-1)
    expanded_masks2.append(e_mask)

    # Apply mask to the image
    m_image = image * e_mask
    edges.append(m_image)

    # Plot the masked image
    plt.figure(figsize=(10, 10))
    plt.imshow(m_image)
    # Place the mask on the image
    show_mask(mask, plt.gca())
    # Place the token on the image
    show_points(input_point1, input_label1, plt.gca())
    plt.title(f'Step 2 Mask {i+1}, Score: {score:.3f}', fontsize=18)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"zstep2_{i}.png"))
    plt.clf()
    plt.imshow(mask)
    plt.title(f'Step 2 Mask {i+1}, Score: {score:.3f}', fontsize=18)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"zstep2_m_{i}.png"))
