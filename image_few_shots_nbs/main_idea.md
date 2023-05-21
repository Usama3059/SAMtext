
## Overview
This portion of the project is to implement the SAM mask using few shots images 

## Idea/Plan
1) Firstly SAM will applied on all images given in the dataset( initial target is 50-80 images )
2) After extracting the mask of all images, these are then cropped from the image.
3) Then, create embedding using  SwimTransformer/ VIT / SAM VIT of cropped mask images.
4) Cluster these embeddings and then select few shots mask images as base images to select the cluster.
5) Then find the close relevant images using cosine similarity and hence preserve the mask of these images only


## Why 

As experienced by the notebooks(which includes CLIP and SAM), its works well with general text descriptions but sometime it is difficult to gain good results with very specific text descriptions, so therefore few shots images is worth experimenting

