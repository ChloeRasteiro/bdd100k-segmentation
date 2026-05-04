
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader



CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

CLASS_NAMES = list(CLASSES.values())
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255


PALETTE = [
    (128, 64, 128),   # road
    (244, 35, 232),   # sidewalk
    (70,  70,  70),   # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170,  30),  # traffic light
    (220, 220,   0),  # traffic sign
    (107, 142,  35),  # vegetation
    (152, 251, 152),  # terrain
    (70,  130, 180),  # sky
    (220,  20,  60),  # person
    (255,   0,   0),  # rider
    (0,    0,  142),  # car
    (0,    0,   70),  # truck
    (0,   60,  100),  # bus
    (0,   80,  100),  # train
    (0,    0,  230),  # motorcycle
    (119,  11,  32),  # bicycle
]


class BDDDataset(Dataset):
    """
    BDD100K semantic segmentation dataset.

    Expected directory layout:
        root/
        ├── images/
        │   ├── train/   *.jpg
        │   └── val/     *.jpg
        └── labels/
            ├── train/   *.png  
            └── val/     *.png
    """

    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

        assert len(self.images) == len(self.masks), (
            f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}"
        )

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return (
            f"BDDDataset | {len(self)} samples | "
            f"images: {self.images_dir}"
        )

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        
        remapped = np.full_like(mask, IGNORE_INDEX, dtype=np.int64)
        for class_id in CLASSES:
            remapped[mask == class_id] = class_id
        return remapped

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = Image.open(mask_path)
        mask = mask.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        mask = np.array(mask)
        mask = self._remap_mask(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()

        return image, mask


def make_dataloaders(
    data_root: str,
    train_transform,
    val_transform,
    batch_size: int = 8,
    num_workers: int = 4,
):
    
    train_ds = BDDDataset(
        images_dir=f"{data_root}/images/train",
        masks_dir=f"{data_root}/labels/train",
        transform=train_transform,
    )
    val_ds = BDDDataset(
        images_dir=f"{data_root}/images/val",
        masks_dir=f"{data_root}/labels/val",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,       
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
    )

    print(train_ds)
    print(val_ds)
    return train_loader, val_loader, train_ds, val_ds
