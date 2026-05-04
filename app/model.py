import torch
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms
import numpy as np
from data.dataset import PALETTE, CLASS_NAMES, NUM_CLASSES
from data.transforms import get_val_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORM = get_val_transform(image_size=512)

from huggingface_hub import hf_hub_download

def load_model():
    path = hf_hub_download(
        repo_id="chloerasteiro/bdd100k-model",
        filename="model/best_dice_focal_bundle.pth",
        repo_type="dataset"
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    config = checkpoint["config"]

    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["data"]["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    return model

def predict(image: Image.Image) -> Image.Image:
    original_size = image.size

    
    img_array = np.array(image.convert("RGB"))
    result = TRANSFORM(image=img_array)
    tensor = result["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = MODEL(tensor)
        mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

    
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(PALETTE):
        colored[mask == class_idx] = color

    return Image.fromarray(colored).resize(original_size, Image.NEAREST)


MODEL = load_model()
