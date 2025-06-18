import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ConvNeXt_Base_Weights
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CnnEncoder:
    """Encodes images into feature embeddings using a pre-trained CNN model.

    Attributes:
        model_name (str): Name of the CNN model to use ('ResNet18' or 'ConvNeXt').
        device (torch.device): Device to run the model on.
    """
    def __init__(
        self,
        model_name: str = "ResNet18",
        device: Optional[torch.device] = None
    ) -> None:
        """Initializes the CnnEncoder with a specified model and device.

        Args:
            model_name (str, optional): Name of the CNN model ('ResNet18' or 'ConvNeXt'). Defaults to 'ResNet18'.
            device (Optional[torch.device], optional): Device to use. If None, uses CUDA if available. Defaults to None.
        """
        self.model_name: str = model_name
        self.device: torch.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_model: torch.nn.Module
        self.transform: transforms.Compose
        self.embedding_size: int
        self.encoder_model, self.transform, self.embedding_size = self.load_model_and_transform()

    def load_model_and_transform(self) -> Tuple[torch.nn.Module, transforms.Compose, int]:
        """Load a pre-trained ResNet or ConvNeXt model and its associated image transformation.

        Returns:
            Tuple[torch.nn.Module, transforms.Compose, int]:
                - The CNN model with the final layer removed.
                - The image transformation pipeline.
                - The embedding size.

        Raises:
            ValueError: If an unsupported model_name is provided.
            Exception: If model weights cannot be loaded.
        """
        try:
            if self.model_name == "ResNet18":
                embedding_size: int = 512
                model: torch.nn.Module = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                transform: transforms.Compose = ResNet18_Weights.DEFAULT.transforms()
            elif self.model_name == "ConvNeXt":
                embedding_size: int = 1024
                model: torch.nn.Module = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
                transform: transforms.Compose = ConvNeXt_Base_Weights.DEFAULT.transforms()
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}. Internet connection may be required.")
            raise
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        return model, transform, embedding_size

    def create_embedding(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """Creates a normalized embedding from an input image.

        Args:
            image (np.ndarray | PIL.Image.Image): Input image as a NumPy array (H, W, C) or PIL Image.

        Returns:
            np.ndarray: Normalized embedding vector.

        Raises:
            ValueError: If the input image shape is invalid.
        """
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                if image.ndim != 3 or image.shape[-1] not in [1, 3]:
                    raise ValueError("Expected shape (H, W, C) with C in [1, 3]")
                image = Image.fromarray(image)  # Convert to PIL
            input_tensor: torch.Tensor = self.transform(image).unsqueeze(0)
            embedding: np.ndarray = self.encoder_model(input_tensor).squeeze().numpy()
            embedding = embedding / np.linalg.norm(embedding)
        return embedding
