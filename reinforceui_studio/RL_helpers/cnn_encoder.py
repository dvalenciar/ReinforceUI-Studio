import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CnnEncoder:
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        model_name: str = "resnet18",
        weights: Optional[models.ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1,
        device: Optional[torch.device] = None
    ) -> None:

        self.image_size = image_size
        self.model_name = model_name
        self.weights = weights
        self.device = device

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_model, self.transform = self.load_model_and_transform()

    def load_model_and_transform(self)-> Tuple[torch.nn.Module, transforms.Compose]:
        """Load a pre-trained ResNet model and the corresponding image transformation.

        Returns:
            Tuple[torch.nn.Module, transforms.Compose]: The model and image transformation.
        """
        try:
            if self.model_name == "resnet18":
                model = models.resnet18(weights=self.weights)
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}. Internet connection may be required.")
            raise

        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return model, transform

    def create_embedding(self, image) -> torch.Tensor:
        with torch.no_grad():
            input_tensor = self.transform(image).unsqueeze(0)
            embedding = self.encoder_model(input_tensor).squeeze().numpy()
            embedding = embedding / np.linalg.norm(embedding)
        return embedding


# def load_model_and_transform(
#     image_size: Tuple[int, int] = (160, 160),
#     model_name: str = "resnet18",
#     weights: Optional[models.ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1,
#     device: Optional[torch.device] = None
# ) -> Tuple[torch.nn.Module, transforms.Compose]:
#     """
#     Load a pre-trained ResNet model and the corresponding image transformation.
#
#     Args:
#         image_size (Tuple[int, int]): The size to which images will be resized.
#         model_name (str): The name of the ResNet model to load.
#         weights: The weights to use for the model.
#         device: The device to load the model onto.
#
#     Returns:
#         Tuple[torch.nn.Module, transforms.Compose]: The model and image transformation.
#     """
#
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     try:
#         if model_name == "resnet18":
#             model = models.resnet18(weights=weights)
#         else:
#             raise ValueError(f"Unsupported model_name: {model_name}")
#     except Exception as e:
#         logger.error(f"Error loading model weights: {e}. Internet connection may be required.")
#         raise
#
#     model.fc = torch.nn.Identity()
#     model.eval()
#     model.to(device)
#     transform = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return model, transform


# def create_embedding(image) -> torch.Tensor:
#     with torch.no_grad():
#         input_tensor = transform(image).unsqueeze(0)
#         embedding = encoder_model(input_tensor).squeeze().numpy()
#         embedding = embedding / np.linalg.norm(embedding)
#     return embedding