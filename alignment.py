"""
Alignment & Token Grouping - Clusters overlapping bounding boxes and groups text candidates
"""
import re
import unicodedata
from typing import List, Dict
from collections import defaultdict

from models import OCRResult, BoundingBox, TextCandidate


class TextAlignment:
    """Handles alignment and grouping of OCR results across models"""
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        Initialize text alignment
        
        Args:
            iou_threshold: IoU threshold for considering bounding boxes as overlapping (0.0 to 1.0)
        """
        self.iou_threshold = iou_threshold
    
    def group_by_region(self, ocr_results: List[OCRResult]) -> List[TextCandidate]:
        """
        Group OCR results by spatial regions (overlapping bounding boxes)
        
        Args:
            ocr_results: List of OCRResult objects from all models
            
        Returns:
            List of TextCandidate objects, each containing candidates for a region
        """
        if not ocr_results:
            return []
        
        # Cluster overlapping bounding boxes
        clusters = self._cluster_bboxes([r.bbox for r in ocr_results])
        
        # Group results by cluster
        text_candidates = []
        for cluster_indices in clusters:
            cluster_results = [ocr_results[i] for i in cluster_indices]
            
            # Merge bounding boxes in cluster to create region bbox
            region_bbox = self._merge_bboxes([r.bbox for r in cluster_results])
            
            # Normalize text in candidates
            normalized_results = []
            for result in cluster_results:
                normalized_text = self._normalize_text(result.text)
                normalized_result = OCRResult(
                    text=normalized_text,
                    bbox=result.bbox,
                    model_name=result.model_name,
                    confidence=result.confidence
                )
                normalized_results.append(normalized_result)
            
            text_candidates.append(TextCandidate(
                candidates=normalized_results,
                region_bbox=region_bbox
            ))
        
        return text_candidates
    
    def _cluster_bboxes(self, bboxes: List[BoundingBox]) -> List[List[int]]:
        """
        Cluster overlapping bounding boxes using IoU
        
        Returns list of clusters, where each cluster is a list of bbox indices
        """
        n = len(bboxes)
        if n == 0:
            return []
        
        # Build adjacency matrix based on IoU
        clusters = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Start new cluster with bbox i
            cluster = [i]
            assigned.add(i)
            
            # Find all overlapping bboxes
            to_check = [i]
            while to_check:
                current_idx = to_check.pop()
                current_bbox = bboxes[current_idx]
                
                for j in range(n):
                    if j in assigned or j == current_idx:
                        continue
                    
                    if current_bbox.iou(bboxes[j]) >= self.iou_threshold:
                        cluster.append(j)
                        assigned.add(j)
                        to_check.append(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _merge_bboxes(self, bboxes: List[BoundingBox]) -> BoundingBox:
        """Merge multiple bounding boxes into one"""
        if not bboxes:
            raise ValueError("Cannot merge empty list of bounding boxes")
        
        x1 = min(bbox.x1 for bbox in bboxes)
        y1 = min(bbox.y1 for bbox in bboxes)
        x2 = max(bbox.x2 for bbox in bboxes)
        y2 = max(bbox.y2 for bbox in bboxes)
        
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison:
        - Unicode normalization (NFD to NFC)
        - Whitespace normalization
        - Remove extra spaces
        """
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_unique_candidates(self, text_candidate: TextCandidate) -> List[tuple]:
        """
        Get unique text candidates with their model names and confidence scores
        
        Returns list of (text, [model_names], avg_confidence) tuples
        """
        # Group by normalized text
        text_groups = defaultdict(lambda: {"models": [], "confidences": []})
        
        for candidate in text_candidate.candidates:
            text_groups[candidate.text]["models"].append(candidate.model_name)
            if candidate.confidence is not None:
                text_groups[candidate.text]["confidences"].append(candidate.confidence)
        
        unique_candidates = []
        for text, data in text_groups.items():
            avg_conf = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else None
            unique_candidates.append((text, data["models"], avg_conf or 0.0))
        
        # Sort by confidence (descending)
        unique_candidates.sort(key=lambda x: x[2], reverse=True)
        
        return unique_candidates
