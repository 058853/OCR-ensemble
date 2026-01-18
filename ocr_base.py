"""
Base OCR interface and abstract class
"""
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import numpy as np

from models import OCRResult, ModelName


class OCRBase(ABC):
    """Base class for all OCR model implementations"""
    
    def __init__(self, model_name: ModelName):
        self.model_name = model_name.value
    
    @abstractmethod
    def process(self, image: Image.Image) -> List[OCRResult]:
        """
        Process an image and return OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            List of OCRResult objects
        """
        pass
    
    def _image_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array"""
        return np.array(image)
