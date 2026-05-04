
import torch
import numpy as np
from data.dataset import CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX


def build_confusion_matrix(model, dataloader, device, forward_fn=None):
    
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks  = masks.long()          

            if forward_fn is not None:
                logits = forward_fn(model, images)
            else:
                logits = model(images)

            
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits.float(),
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            preds = logits.argmax(dim=1).cpu().numpy()   
            masks = masks.numpy()                         

            for pred, mask in zip(preds, masks):
                valid = mask != IGNORE_INDEX
                pred_v = pred[valid]
                mask_v = mask[valid]
                
                np.add.at(conf, (mask_v, pred_v), 1)

    return conf


def iou_from_confusion(conf: np.ndarray):
    
    iou_per_class = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp  
        fn = conf[c, :].sum() - tp   
        denom = tp + fp + fn
        iou_per_class.append(tp / denom if denom > 0 else 0.0)
    return np.array(iou_per_class)


def pixel_accuracy_from_confusion(conf: np.ndarray) -> float:
    return conf.diagonal().sum() / conf.sum()


def evaluate(model, dataloader, device, forward_fn=None) -> dict:
    
    conf = build_confusion_matrix(model, dataloader, device, forward_fn)
    iou  = iou_from_confusion(conf)
    acc  = pixel_accuracy_from_confusion(conf)
    miou = iou.mean()

    print("\n" + "─" * 40)
    print(f"{'Class':<16} {'IoU':>8}")
    print("─" * 40)
    for name, val in zip(CLASS_NAMES, iou):
        print(f"  {name:<14} {val:>8.3f}")
    print("─" * 40)
    print(f"  {'mIoU':<14} {miou:>8.3f}")
    print(f"  {'Pixel acc.':<14} {acc:>8.3f}")
    print("─" * 40 + "\n")

    return {
        "iou_per_class": iou,
        "miou": miou,
        "pixel_accuracy": acc,
        "confusion_matrix": conf,
        "class_names": CLASS_NAMES,
    }
