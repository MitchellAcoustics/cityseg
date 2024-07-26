# %%
from transformers import pipeline
import cv2
from cityscape_seg.palettes import ADE20K_PALETTE
import matplotlib.pyplot as plt
import numpy as np

file = "/Users/mitch/Documents/GitHub/cityscape-seg/example_inputs/EustonTap-Screenshot1.png"
image = cv2.imread(file)

pipe = pipeline(
    model="microsoft/beit-large-finetuned-ade-640-640", task="image-segmentation"
)
res = pipe(file)
# res[0]['mask'].show()

# %%

color_seg = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
palette = np.array(ADE20K_PALETTE)

# %%

label2id = pipe.model.config.label2id

for mask in res:
    pred_seg = np.array(mask["mask"])
    for label, color in enumerate(palette):
        color_seg[pred_seg != 0, :] = palette[
            label2id[mask["label"]]
        ]  # color the segmentation map

color_seg = color_seg[..., ::-1]  # convert to BGR

# %%
img = (
    np.array(image) * 0.5 + color_seg * 0.5
)  # plot the image with the segmentation map
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()
