# BDD100K Semantic Segmentation

Implementing U-Net with EfficientNet-B3 for semantic segmentation on the challenging BDD100K autonomous driving dataset.

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

| ID | Class | Class Distribution (%) |
|----|-------|------------------------|
| 0 | Road | 24.976 |
| 1 | Sidewalk | 2.371 |
| 2 | Building | 17.238 |
| 3 | Wall | 0.412 |
| 4 | Fence | 0.932 |
| 5 | Pole | 1.122 | 
| 6 | Traffic Light | 0.157|
| 7 | Traffic Sign | 0.267 |
| 8 | Vegetation | 17.870 |
| 9 | Terrain | 1.054 | 
|10 | Sky | 20.866 |
| 11 | Person | 0.286 | 
| 12 | Rider | 0.011 | 
| 13 | Car |  10.476 | 
| 14 | Truck |  1.176 |
| 15 | Bus | 0.728 |
| 16 | Train | 0.014 | 
| 17 | Motorcycle | 0.027 |
| 18 | Bicycle | 0.018 |

* Top 5 classes : Road, Sky, Vegetation, Building, Car
* Bottom 5 classes: Rider, Train, Bicycle, Motorcycle, Traffic Light

## Model Architecture

U-Net with an EfficientNet-B3 encoder.



* Encoder  : EfficientNet-B3 pretrained on ImageNet
* Decoder : U-Net style decoder with skip connections
* Input Size : 640x640x3
* Output Size : 640x640x19

Skip connections preserve spatial information crucial for segmentation
Combines low-level details (edges, textures) with high-level semantics (objects)

##Training Process

|  | Encoder | Decoder | Segmentation Head|
|--|---------|---------|------------------|
|Phase 1| Frozen| lr: 1×10⁻³|lr: 1×10⁻³|
|Phase 2| lr :1×10⁻⁵ | lr: 5×10⁻⁵ | lr: 5×10⁻⁵ |

Phase 1: Warm-up with frozen encoder to learn segmentation task 
Phase 2: Fine-tune entire network for optimal performance. 


**Loss Function**

* Dice Loss : Optimize overlap
* Focal Loss : Handles class imbalance by focusing on hard examples
* CrossEntropy Loss : Pixel-wise classification

**Optimization**
-Optimizer : Adam
-Scheduler : Cosine Annealing (T\_max=30, η\_min=1×10⁻⁷)
-Regularization: Early Stopping, data augmentation


## Results

IoU per class:

|Class | IoU | 
|------|-----|
|road  |    0.776|
|sidewalk    |0.428|
|building    |0.717|
|wall        |0.128|
|fence       |0.267|
|pole        |0.313|
|traffic light |0.271|
|traffic sign |0.220|
|vegetation  |0.787|
|terrain     |0.293|
|sky         |0.920|
|person      |0.348|
|rider       |0.128|
|car         |0.715|
|truck       |0.324|
|bus         |0.338|
|train       |0.000|
|motorcycle  |0.174|
|bicycle     |0.036|

Mean IoU: 0.378
Best Validation Loss:0.470

