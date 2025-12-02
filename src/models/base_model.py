from abc import ABC, abstractmethod
import torch
import numpy as np
from PIL import Image
from loguru import logger
from typing import Union, Dict, Any

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.project.device)
        self.model = None

        logger.info(f"Initializing {self.__class__.__name__} on {self.device}...")
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load weights from Hugging Face."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Run inference.
        Args:
            image (PIL.Image): Input image.
        Returns:
            Union[np.ndarray, Dict]: Output mask, depth map, or dictionary of candidates.
        """
        pass