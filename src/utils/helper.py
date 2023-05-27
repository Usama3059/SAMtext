import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torch 

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, segmentation_mask,path_save,mask_c):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    black_image.save(f"{path_save}/{mask_c}.jpg")
    return black_image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
    
@torch.no_grad()
def retrieve(elements: list[Image.Image], search_image: list[Image.Image],model,preprocess,device) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]

    stacked_images = torch.stack(preprocessed_images)
    image_features_init = model.encode_image(stacked_images)
    preprocessed_images = [preprocess(image).to(device) for image in search_image]

    stacked_images = torch.stack(preprocessed_images)   
    search_features_init = model.encode_image(stacked_images)
    print(image_features_init.shape,search_features_init.shape)

    image_features = image_features_init / image_features_init.norm(dim=-1, keepdim=True) 
    search_features = search_features_init / search_features_init.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ search_features.T
    return probs[:, 0].softmax(dim=0),image_features_init,search_features_init

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]