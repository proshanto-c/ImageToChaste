from import_modules import *

matplotlib.use("Agg")

checkpoint = "sam/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
point_or_box = "point"

print("Loading image...")
image = Image.open("data/CLAHE_output/000.png")
image = np.array(image.convert("RGB"))
# image.shape is formatted (y, x, c) {c is the number of colours - in this case, c=3} **Figure out why**
y, x, _ = image.shape
x2 = round(x/2)
y2 = round(y/2)
input_point1 = np.array([[x2,y2]])
input_label1 = np.array([1])
input_box = np.array([x2-25, y2-25, x2+25, y2+25])

input_point2 = np.array([[20, 20]])
input_label2 = np.array([1])

# First prediction (isolate the scan from the background)
print("Generating first prediction...")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    print(image.shape)
    predictor.set_image(image)
    imasks, iscores, ilogits = predictor.predict(point_coords=input_point1, point_labels=input_label1, multimask_output=True)
    # Expand the mask to being a 3 dimensional object
    new_mask = np.expand_dims(imasks[2], axis=-1)
    new_mask = np.repeat(new_mask, 3, axis=-1)
    print(new_mask.shape)
    image2 = np.multiply(image, new_mask)
    print(image2.shape)
    predictor.set_image(image2)
    masks, scores, logits = predictor.predict(point_coords=input_point2, point_labels=input_label2, multimask_output=True)


print("Plotting and saving masks...")
if point_or_box == "point":
    for i, (mask, score) in enumerate(zip(masks, scores)):    
        plt.figure(figsize=(10,10))
        plt.imshow(image2)
        show_mask(mask, plt.gca())
        show_points(input_point2, input_label2, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"outs/2s_test_{i}.png")
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