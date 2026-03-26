import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image

try:
    import clip as openai_clip
except ImportError:
    openai_clip = None

try:
    import open_clip
except ImportError:
    open_clip = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid"
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    return model

# CLIP for roof classification
roof_classes = ["RCC roof", "tiled roof", "tin roof", "other roof material"]

clip_model = None
clip_preprocess = None
clip_tokenize = None
clip_backend = None

def _ensure_clip_loaded():
    global clip_model, clip_preprocess, clip_tokenize, clip_backend

    if clip_model is not None and clip_preprocess is not None and clip_tokenize is not None:
        return True

    try:
        if openai_clip is not None and hasattr(openai_clip, "load") and hasattr(openai_clip, "tokenize"):
            clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=device)
            clip_tokenize = openai_clip.tokenize
            clip_backend = "openai_clip"
            return True

        if open_clip is not None:
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device=device,
            )
            clip_tokenize = open_clip.get_tokenizer("ViT-B-32")
            clip_backend = "open_clip"
            return True
    except Exception:
        clip_model = None
        clip_preprocess = None
        clip_tokenize = None
        clip_backend = None

    return False

def classify_roof(patch: np.ndarray):
    if not _ensure_clip_loaded():
        return "unknown roof material"

    image = Image.fromarray(patch)
    image = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        text = clip_tokenize(roof_classes)
        if clip_backend == "openai_clip":
            text = text.to(device)
        logits_per_image, _ = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1)
    return roof_classes[probs.argmax().item()]