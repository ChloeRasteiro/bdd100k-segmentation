# BDD100K Semantic Segmentation

Implementing U-Net with EfficientNet-B3 for semantic segmentation on the challenging BDD100K autonomous driving dataset.

---

## Dataset

The BDD100K (Berkeley DeepDrive) dataset for image segmentation consists of:

* 7,000 images for training
* 1,000 images for validation
* 2,000 images for testing

The images contain diverse driving scenarios (city, highway, day, night, rain, etc.), which makes the dataset complex and challenging.

Image specifications:

* Original resolution: 1280×720 pixels
* Training resolution: 640×640 pixels (resized)
* 19 semantic classes



The dataset is highly imbalanced which is typical for real-world driving scenes:

| ID  | Class         | Class Distribution (%) |
| --- | ------------- | ---------------------- |
| 0   | Road          | 24.976                 |
| 1   | Sidewalk      | 2.371                  |
| 2   | Building      | 17.238                 |
| 3   | Wall          | 0.412                  |
| 4   | Fence         | 0.932                  |
| 5   | Pole          | 1.122                  |
| 6   | Traffic Light | 0.157                  |
| 7   | Traffic Sign  | 0.267                  |
| 8   | Vegetation    | 17.870                 |
| 9   | Terrain       | 1.054                  |
| 10  | Sky           | 20.866                 |
| 11  | Person        | 0.286                  |
| 12  | Rider         | 0.011                  |
| 13  | Car           | 10.476                 |
| 14  | Truck         | 1.176                  |
| 15  | Bus           | 0.728                  |
| 16  | Train         | 0.014                  |
| 17  | Motorcycle    | 0.027                  |
| 18  | Bicycle       | 0.018                  |


* Top 5 classes : Road, Sky, Vegetation, Building, Car
* Bottom 5 classes: Rider, Train, Bicycle, Motorcycle, Traffic Light

---

## Model Architecture

U-Net with an EfficientNet-B3 encoder.

Architecture Details:  
* Encoder  : EfficientNet-B3 pretrained on ImageNet
* Decoder : U-Net style decoder with skip connections
* Input Size : 640x640x3
* Output Size : 640x640x19

Skip connections preserve spatial information crucial for segmentation by combining low-level details (edges, textures) with high-level semantics (objects).

---

## Training Process

Two-Phase Training Strategy

|  | Encoder | Decoder | Segmentation Head|
|--|---------|---------|------------------|
|Phase 1| Frozen| lr: 1×10⁻³|lr: 1×10⁻³|
|Phase 2| lr :1×10⁻⁵ | lr: 5×10⁻⁵ | lr: 5×10⁻⁵ |

Phase 1: Warm-up with frozen encoder to learn segmentation task 
Phase 2: Fine-tune entire network for optimal performance. 

Training Configuration
 - Batch Size : 8 
 - Epochs : Phase 1: 50 , Phase 2 : 30
 - Data Augmentation:
	 - Random horizontal flip (p=0.5)
	 - Random brightness/contrast (p=0.3)
	 - Gaussian noise (p=0.2)
	 - Blur (limit=3, p=0.2)

**Loss Function**
Combined loss approach to handle multiple challenfes:  

* Dice Loss (weight:0.25) : Optimize overlap between prediction and ground truth
* Focal Loss (weight:0.5) : Handles class imbalance by focusing on hard examples
* CrossEntropy Loss weight (weight:0.25) : Pixel-wise classification

**Total Loss** = 0.5 × Focal +0.25 x Dice + 0.25 × CrossEntropy

**Optimization**

* Optimizer : Adam
* Scheduler : ReduceLROnPlateau
* Regularization : Early Stopping, Data Augmentation

---

## Results

### Performance
* Mean IoU : 0.430
* Best Validation Loss : 0.2632

### Training Curves


### IoU per class

| Class      | IoU   | Distribution (%) | Class         | IoU   | Distribution (%) |     |
| ---------- | ----- | ---------------- | ------------- | ----- | ---------------- | --- |
| Sky        | 0.921 | 20.866           | Terrain       | 0.348 | 1.054            |     |
| Vegetation | 0.797 | 17.870           | Traffic Light | 0.401 | 0.157            |     |
| Road       | 0.728 | 24.976           | Fence         | 0.280 | 0.932            |     |
| Building   | 0.739 | 17.238           | Traffic Sign  | 0.290 | 0.267            |     |
| Car        | 0.762 | 10.476           | Motorcycle    | 0.269 | 0.027            |     |
| Sidewalk   | 0.477 | 2.371            | Wall          | 0.163 | 0.412            |     |
| Person     | 0.530 | 0.286            | Rider         | 0.186 | 0.011            |     |
| Bus        | 0.520 | 0.728            | Bicycle       | 0.049 | 0.018            |     |
| Truck      | 0.354 | 1.176            | Train         | 0.000 | 0.014            |     |
| Pole       | 0.357 | 1.122            |               |       |                  |     |




---

##  Analysis 

Correlation observation:
There is a clear correlation between class frequency and performance. 
Classes with >10% distribution (Road, Sky, Vegetation, Building, Car) all achieve >0.70 IoU. Meanwhile, classes representing <0.1% of the dataset (Train,  Bicycle, Rider) fail almost entirely with <0.2 IoU. 
This demonstrates how extreme class imbalance directly impacts model performance.

### Strengths 
**Large, frequent objects:** 
- Sky (0.921), Vegetation (0.797), Road (0.728): Nearly perfect segmentation 

These classes benefit from large pixel coverage, high frequency in the training data, and consistent visual appearance, making them easier to detect.


### Limitations 
**Small objects:** 
- Bicycles (0.049), Traffic lights (0.401): Too small at 640×640 resolution 

These small objects are too difficult to detect at 640×640 resolution. Fine details are lost during resizing from the original 1280×720, making precise segmentation nearly impossible.

**Rare classes:** 
- Train (0.000), Rider (0.186): Only 0.01% of dataset each 

These classes represent only 0.01% of the dataset each, providing insufficient training examples for the model to learn their characteristics. The model has seen fewer than 100 examples during the entire training process.

