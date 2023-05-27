# SAMtext


## Updates
- [completed] Selective search based annotations using zero shot text prompt.
- [completed] Patch and GradCAM based annotations using zero shot text prompt.
- [completed] Basic idea implementation of zero shot images annotations.
- [working]Module implementation to annotation upto 100k images with few shots images.



## Using Zero shot image embeddings

### Overview
This portion of the project is to implement the SAM mask using few shots images 

### Idea/Plan
1) Firstly SAM will applied on all images given in the dataset( initial target is 50-80 images )
2) After extracting the mask of all images, these are then cropped from the image.
3) Then, create embedding using  SwimTransformer/ VIT / SAM VIT of cropped mask images.
4) Cluster these embeddings and then select few shots mask images as base images to select the cluster.
5) Then find the close relevant images using cosine similarity and hence preserve the mask of these images only

### Basic implementation

[notebook image zero shots](image_few_shots_nbs/main_idea.ipynb)

### Why 

As experienced by the notebooks(which includes CLIP and SAM), its works well with general text descriptions but sometime it is difficult to gain good results with very specific text descriptions, so therefore few shots images is worth experimenting


## Using Zero shot text embeddings

### Overview
This portion of the project is to implement the SAM mask predictions using zero shots text prompt.


Implementation of  text prompt-controlled segmentation using selective search and CLIP can be found in [notebook clip text zero shots v1](text_zero_shots_nbs/using_clip/01_CLIP_text-prompt_using_SelectiveSearch.ipynb)

Working on segmenting small objects(based on patch size value) with low-quality image descriptions, such as car blinkers. I have used a sliding window technique to extract image patches, which were then subjected to clip retrieval, CLIP Grad-CAM, then point extraction and then SAM to get the segmentation results can be found in [notebook clip text zero shots v2](text_zero_shots_nbs/using_clip/02_CLIP_text_prompt_using_patches_GradCAM.ipynb)




