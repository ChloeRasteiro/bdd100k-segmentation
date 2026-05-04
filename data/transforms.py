
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(image_size: int = 512) -> A.Compose:
    
    return A.Compose([
        A.Resize(int(image_size * 1.1), int(image_size * 1.1)),   
        A.RandomCrop(image_size, image_size),                      
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
        A.RandomRain(drop_length=10, p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 512) -> A.Compose:
    
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def denormalize(tensor):
    
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
