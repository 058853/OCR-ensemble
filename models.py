"""
Common data models and schemas for OCR outputs
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelName(str, Enum):
    """Supported OCR model names"""
    TESSERACT = "tesseract"
    PADDLE_OCR = "paddleocr"
    SURYA_OCR = "surya_ocr"
    DEEPSEEK_OCR = "deepseek_ocr"


@dataclass
class BoundingBox:
    """Bounding box coordinates (x1, y1, x2, y2)"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BoundingBox':
        return cls(**data)
    
    def area(self) -> float:
        """Calculate bounding box area"""
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union (IoU) with another bounding box"""
        # Calculate intersection
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union = self.area() + other.area() - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


@dataclass
class OCRResult:
    """Normalized OCR result from a single model"""
    text: str
    bbox: BoundingBox
    model_name: str
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "model_name": self.model_name,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRResult':
        return cls(
            text=data["text"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            model_name=data["model_name"],
            confidence=data.get("confidence")
        )


@dataclass
class TextCandidate:
    """Multiple text candidates for the same region"""
    candidates: List[OCRResult]
    region_bbox: BoundingBox  # Merged bounding box for the region
    
    def get_candidates_with_conf(self) -> List[tuple]:
        """Get (text, model_name, confidence) tuples"""
        return [
            (c.text, c.model_name, c.confidence or 0.0)
            for c in self.candidates
        ]


@dataclass
class ReconciledResult:
    """Final reconciled OCR result after LLM voting"""
    text: str
    bbox: BoundingBox
    confidence: float
    reason: Optional[str] = None
    source_models: List[str] = None
    
    def __post_init__(self):
        if self.source_models is None:
            self.source_models = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "reason": self.reason,
            "source_models": self.source_models
        }
