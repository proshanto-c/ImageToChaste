from import_modules import *

matplotlib.use("Agg")

checkpoint = "sam/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
point_or_box = "box"

print("Loading image...")
image = Image.open("data/INV_CLAHE_1/000.png")
image = np.array(image.convert("RGB"))
# image.shape is formatted (y, x, c) {c is the number of colours - in this case, c=3} **Figure out why**
y, x, _ = image.shape
x2 = round(x/2)
y2 = round(y/2)
input_point = np.array([[x2,y2]])
input_label = np.array([1])
input_box = np.array([x2-25, y2-25, x2+25, y2+25])

print("Generating predictions...")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    if point_or_box == "point":
        masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
    else:
        masks, scores, logits = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=True)

print("Plotting and saving masks...")
if point_or_box == "point":
    for i, (mask, score) in enumerate(zip(masks, scores)):    
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"outs/INV_test_{i}.png")
else:
    show_masks(image, masks, scores, box_coords=input_box)

plt.close()

'''
Model takes as input an image and a point, then outputs 3 different masks with scores. The scores represent 
confidence values in the mask predictions. For the initial image, higher score largely represents a bigger mask,
with smaller scores representing big cells, and very high scores (>0.9) overshooting the scan. What needs to be
tested:
- What is the format of the masks that come out as a result? Can they be passed back into the predictor?
- What is the general score-mask relationship when the point is centered?'''