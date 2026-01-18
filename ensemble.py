"""
OCR Ensemble Layer - Runs multiple OCR models in parallel and normalizes outputs
"""
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from ocr_base import OCRBase
from models import OCRResult, ModelName


class OCREnsemble:
    """Manages multiple OCR models and runs them in parallel"""
    
    def __init__(self, models: List[OCRBase]):
        """
        Initialize ensemble with OCR models
        
        Args:
            models: List of OCRBase instances
        """
        self.models = models
        if not models:
            raise ValueError("At least one OCR model must be provided")
    
    def process(self, image: Image.Image, parallel: bool = True, max_workers: Optional[int] = None) -> List[OCRResult]:
        """
        Process image with all models in the ensemble
        
        Args:
            image: PIL Image to process
            parallel: Whether to run models in parallel
            max_workers: Maximum number of parallel workers (None = auto)
            
        Returns:
            List of normalized OCRResult objects from all models
        """
        if parallel and len(self.models) > 1:
            return self._process_parallel(image, max_workers)
        else:
            return self._process_sequential(image)
    
    def _process_parallel(self, image: Image.Image, max_workers: Optional[int]) -> List[OCRResult]:
        """Process image with all models in parallel"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(model.process, image): model 
                for model in self.models
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error processing with {model.model_name}: {e}")
                    # Continue with other models even if one fails
        
        return all_results
    
    def _process_sequential(self, image: Image.Image) -> List[OCRResult]:
        """Process image with all models sequentially"""
        all_results = []
        
        for model in self.models:
            try:
                results = model.process(image)
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing with {model.model_name}: {e}")
                # Continue with other models even if one fails
        
        return all_results
    
    def add_model(self, model: OCRBase):
        """Add a new OCR model to the ensemble"""
        self.models.append(model)
    
    def remove_model(self, model_name: str):
        """Remove a model from the ensemble by name"""
        self.models = [m for m in self.models if m.model_name != model_name]
