---
title: BDD100K Segmentation
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---




# BDD100K Semantic Segmentation
**End-to-end deployed segmentation pipeline**   
 U-Net + EfficientNet-B3 
trained on BDD100K, served via FastAPI and Gradio on HuggingFace Spaces.

mIoU: 51.6% | Pixel Accuracy: 91.8% | 19 classes | [ Live Demo](#)



##  Live Demo
[Try it here](https://huggingface.co/spaces/ChloeRasteiro/bdd100k-segmentation)

Upload any driving scene image → get semantic segmentation in real-time.
 


## Dataset

The BDD100K (Berkeley DeepDrive) dataset for image segmentation consists of:

* 7,000 images for training
* 1,000 images for validation
* 2,000 images for testing

The images contain diverse driving scenarios (city, highway, day, night, rain, etc.), which makes the dataset complex and challenging.

Image specifications:

* Original resolution: 1280×720 pixels
* Training resolution: 512×512 pixels (resized)
* 19 semantic classes



The dataset is highly imbalanced which is typical for real-world driving scenes:

* Top 5 classes : Road, Sky, Vegetation, Building, Car
* Bottom 5 classes: Rider, Train, Bicycle, Motorcycle, Traffic Light



## Model Architecture

U-Net with an EfficientNet-B3 encoder.

Architecture Details:  
* Encoder  : EfficientNet-B3 pretrained on ImageNet
* Decoder : U-Net style decoder with skip connections
* Input Size : 512x512x3
* Output Size : 512x512x19

Skip connections preserve spatial information crucial for segmentation by combining low-level details (edges, textures) with high-level semantics (objects).



## Training Process

**Two-Phase Training Strategy**

* Phase 1 — frozen encoder :  decoder + head lr = 1×10⁻³  
Warm-up with frozen encoder to learn segmentation task 

* Phase 2 — full fine-tune :  encoder lr = 1×10⁻⁵ / decoder + head lr = 5×10⁻⁵  
Fine-tune entire network for optimal performance.
 

**Training Configuration**
 - Batch Size : 8 
 - Epochs : Phase 1: 50/50 , Phase 2 : 26/50  
 Phase 2 converged early (26/50 epochs): early stopping prevented 
overfitting once validation loss plateaued at 0.2365.
 - Data Augmentations (training): 
    - Random crop after slight oversize (512×512)
    - Horizontal flip, ColorJitter, GaussNoise, Blur
    - RandomFog + RandomRain : specific to BDD100K adverse weather scenes

Validation: Resize + Normalize only.


**Loss Function**
Combined loss approach to handle multiple challenfes:  

* Dice Loss (weight:0.5) : Optimize overlap between prediction and ground truth
* Focal Loss (weight:0.5) : Handles class imbalance by focusing on hard examples

Combined Loss = 0.5 × Focal +0.5 x Dice 

**Optimization**

* Optimizer : Adam
* Scheduler : ReduceLROnPlateau
* Regularization : Early Stopping, Data Augmentation


## Results

### Performance

| Metric          | Score  |
| --------------- | ------ |
| mIoU            | 51.6%  |
| Pixel Accuracy  | 91.8%  |
| Validation Loss | 0.2632 |


### IoU per Class

| Class         | IoU   | Distribution (%) |
| ------------- | ----- | ---------------- |
| Sky           | 0.949 | 20.866           |
| Road          | 0.937 | 24.976           |
| Vegetation    | 0.840 | 17.870           |
| Building      | 0.821 | 17.238           |
| Car           | 0.876 | 10.476           |
| Sidewalk      | 0.602 | 2.371            |
| Person        | 0.623 | 0.286            |
| Bus           | 0.678 | 0.728            |
| Truck         | 0.479 | 1.176            |
| Pole          | 0.395 | 1.122            |
| Terrain       | 0.470 | 1.054            |
| Traffic Light | 0.515 | 0.157            |
| Traffic Sign  | 0.488 | 0.267            |
| Fence         | 0.402 | 0.932            |
| Motorcycle    | 0.309 | 0.027            |
| Wall          | 0.268 | 0.412            |
| Rider         | 0.136 | 0.011            |
| Bicycle       | 0.016 | 0.018            |
| Train         | 0.000 | 0.014            |




##  Analysis 

Correlation observation:
There is a clear correlation between class frequency and performance. 
Classes with >10% distribution (Road, Sky, Vegetation, Building, Car) all achieve >0.70 IoU. Meanwhile, classes representing <0.1% of the dataset (Train,  Bicycle, Rider) fail almost entirely with <0.2 IoU. 
This demonstrates how extreme class imbalance directly impacts model performance.

### Strengths 
**Large, frequent objects:** 
- Sky (0.949), Vegetation (0.840), Road (0.937): Nearly perfect segmentation 

These classes benefit from large pixel coverage, high frequency in the training data, and consistent visual appearance, making them easier to detect.


### Limitations 
**Small objects:** 
- Bicycles (0.049), Traffic lights (0.401): Too small at 512×512 resolution 

These small objects are too difficult to detect at 640×640 resolution. Fine details are lost during resizing from the original 1280×720, making precise segmentation nearly impossible.

**Rare classes:** 
- Train (0.000), Rider (0.186): Only 0.01% of dataset each 
These classes represent only 0.01% of the dataset each, providing insufficient training examples for the model to learn their characteristics. The model has seen fewer than 100 examples during the entire training process.

## Deployment
- **API** : FastAPI REST endpoint (`/segment`)
- **Interface** : Gradio — upload image, visualize segmentation
- **Container** : Docker
- **Hosting** : HuggingFace Spaces

## Stack
`PyTorch` `segmentation-models-pytorch` `FastAPI` `Gradio` `Docker` `Albumentations` `HuggingFace`