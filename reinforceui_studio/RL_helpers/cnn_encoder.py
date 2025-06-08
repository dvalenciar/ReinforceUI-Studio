import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def load_model_and_transform(
    image_size: Tuple[int, int] = (160, 160),
    model_name: str = "resnet18",
    weights: Optional[torchvision.models.ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, transforms.Compose]:
    """
    Load a pre-trained ResNet model and the corresponding image transformation.

    Args:
        image_size (Tuple[int, int]): The size to which images will be resized.
        model_name (str): The name of the ResNet model to load.
        weights: The weights to use for the model.
        device: The device to load the model onto.

    Returns:
        Tuple[torch.nn.Module, transforms.Compose]: The model and image transformation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if model_name == "resnet18":
            model = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}. Internet connection may be required.")
        raise

    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform
