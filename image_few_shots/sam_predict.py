"""

This file mainly contains the Class which use SAM model and create a list of cropped masks of all input images 

"""


import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image, ImageDraw
import clip
import torch
import os 
import numpy as np
import glob 

class SAM_mask():
    def __init__(self,images_folder_path,checkpoint_path  ="sam_vit_h_4b8939.pth" ):
        self.images_folder_path = images_folder_path
        self.checkpoint_path = checkpoint_path
        
        if os.path.exits(self.images_folder_path):
            self.list_images_path = glob.glob(f"{self.images_folder_path}/*")
        else:
            raise Exception("Images folder does not found")
        
        
    
    def generate_mask(self):
        self.allmasks = []
        self.mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=f"{self.checkpoint_path}"))
        for image_path in self.images_folder_path:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imagemasks = self.mask_generator.generate(image)
            
            PILimage = Image.open(image_path)
            cropped_boxes = []

            for mask in imagemasks:
                cropped_boxes.append(self.segment_image(PILimage, mask["segmentation"]).crop(self.convert_box_xywh_to_xyxy(mask["bbox"])))
            self.allmasks.append(cropped_boxes)
            
            
    def convert_box_xywh_to_xyxy(self,box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]
    
    def segment_image(self,image, segmentation_mask):
        image_array = np.array(image)
        segmented_image_array = np.zeros_like(image_array)
        segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new("RGB", image.size, (0, 0, 0))
        transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
        transparency_mask[segmentation_mask] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image
    

        
