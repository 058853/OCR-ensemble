"""
Concrete OCR model implementations
"""
from typing import List, Optional
from PIL import Image
import numpy as np

from ocr_base import OCRBase
from models import OCRResult, BoundingBox, ModelName

# Detection imports (This is likely where your error is)
from surya.model.detection.segformer import (
    load_model as load_det_model, 
    load_processor as load_det_processor
)

# Recognition imports
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

# Main OCR runner
from surya.ocr import run_ocr

# from surya.model.detection.model import (
#     load_model as load_det_model,
#     load_processor as load_det_processor
# )

class TesseractOCR(OCRBase):
    """Tesseract OCR implementation"""
    
    def __init__(self):
        super().__init__(ModelName.TESSERACT)
        try:
            import pytesseract
            self.tesseract = pytesseract
        except ImportError:
            raise ImportError("pytesseract not installed. Install with: pip install pytesseract")
    
    def process(self, image: Image.Image) -> List[OCRResult]:
        """Process image with Tesseract OCR"""
        # Get detailed data with bounding boxes
        data = self.tesseract.image_to_data(image, output_type=self.tesseract.Output.DICT)
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text:  # Skip empty text
                conf = float(data['conf'][i]) if data['conf'][i] != -1 else None
                if conf is not None:
                    conf = conf / 100.0  # Normalize to 0-1
                
                bbox = BoundingBox(
                    x1=float(data['left'][i]),
                    y1=float(data['top'][i]),
                    x2=float(data['left'][i] + data['width'][i]),
                    y2=float(data['top'][i] + data['height'][i])
                )
                
                results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    model_name=self.model_name,
                    confidence=conf
                ))
        
        return results


class PaddleOCRModel(OCRBase):
    """PaddleOCR implementation"""
    
    def __init__(self):
        super().__init__(ModelName.PADDLE_OCR)
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        except ImportError:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
    
    def process(self, image: Image.Image) -> List[OCRResult]:
        """Process image with PaddleOCR"""
        img_array = np.array(image)
        ocr_result = self.ocr.ocr(img_array, cls=True)
        
        results = []
        
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line:
                    # PaddleOCR format: [[bbox], (text, confidence)]
                    bbox_coords = line[0]
                    text, conf = line[1]
                    
                    # Extract bounding box coordinates
                    xs = [point[0] for point in bbox_coords]
                    ys = [point[1] for point in bbox_coords]
                    
                    bbox = BoundingBox(
                        x1=min(xs),
                        y1=min(ys),
                        x2=max(xs),
                        y2=max(ys)
                    )
                    
                    results.append(OCRResult(
                        text=text.strip(),
                        bbox=bbox,
                        model_name=self.model_name,
                        confidence=conf
                    ))
        
        return results


class SuryaOCR(OCRBase):
    """Surya OCR implementation"""
    
    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize Surya OCR
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi']). Default: ['en']
        """
        super().__init__(ModelName.SURYA_OCR)
        self.languages = languages or ['en']
        
        try:
            
            self.run_ocr = run_ocr
            self.load_det_model = load_det_model
            self.load_det_processor = load_det_processor
            self.load_rec_model = load_rec_model
            self.load_rec_processor = load_rec_processor
            
            # Load models (will be cached)
            self.det_processor = None
            self.det_model = None
            self.rec_model = None
            self.rec_processor = None
            
        except ImportError:
            raise ImportError(
                "surya-ocr not installed. Install with: pip install surya-ocr"
            )
    
    def process(self, image: Image.Image) -> List[OCRResult]:
        """Process image with Surya OCR"""
        # Lazy load models on first use
        if self.det_model is None:
            self.det_processor = self.load_det_processor()
            self.det_model = self.load_det_model()
        
        if self.rec_model is None:
            self.rec_model = self.load_rec_model()
            self.rec_processor = self.load_rec_processor()
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        pil_image = Image.fromarray(img_array)
        predictions  = self.run_ocr(
            [pil_image],
            [self.languages],
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor
        )
        
        results = []
        
        if predictions and len(predictions) > 0:
            prediction = predictions[0]  # First (and only) image result
            
            for line in prediction.text_lines:
                # Extract text
                text = line.text.strip()
                if not text:
                    continue
                
                # Extract bounding box (Surya uses polygon format, convert to bbox)
                bbox_polygon = line.bbox
                x1, y1, x2, y2 = line.bbox
                
                bbox = BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                )
                
                # Surya doesn't provide confidence per line, use default or calculate
                confidence = getattr(line, 'confidence', None) or 0.9
                
                results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    model_name=self.model_name,
                    confidence=confidence
                ))
        
        return results


class DeepSeekOCR(OCRBase):
    """DeepSeek OCR implementation using DeepSeek Vision API"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize DeepSeek OCR
        
        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: API base URL (defaults to DeepSeek's official endpoint)
        """
        super().__init__(ModelName.DEEPSEEK_OCR)
        import os
        
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY env var or pass api_key parameter"
            )
        
        # Try OpenAI-compatible client first (DeepSeek uses OpenAI-compatible API)
        try:
            from openai import OpenAI
            self.base_url = base_url or "https://api.deepseek.com"
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.use_openai_format = True
        except ImportError:
            raise ImportError(
                "openai package required for DeepSeek OCR. Install with: pip install openai"
            )
    
    def process(self, image: Image.Image) -> List[OCRResult]:
        """Process image with DeepSeek Vision API"""
        import base64
        from io import BytesIO
        
        # Convert image to base64
        buffered = BytesIO()
        # Save as PNG for better quality
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare prompt for OCR task
        prompt = """Extract all text from this image with accurate bounding box coordinates.
Return the result as JSON with the following format:
{
  "text_regions": [
    {
      "text": "extracted text",
      "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}
    }
  ]
}"""
        
        try:
            # Call DeepSeek Vision API
            response = self.client.chat.completions.create(
                model="deepseek-v2",  # or "deepseek-chat" depending on available models
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            import json
            import re
            
            # Clean response and extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                ocr_data = json.loads(json_match.group())
                return self._parse_deepseek_response(ocr_data, image.size)
            else:
                # Fallback: treat entire response as text (simple OCR)
                # Return as single result with full image bbox
                return [OCRResult(
                    text=response_text.strip(),
                    bbox=BoundingBox(0, 0, image.width, image.height),
                    model_name=self.model_name,
                    confidence=0.8
                )]
                
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return []
    
    def _parse_deepseek_response(self, ocr_data: dict, image_size: tuple) -> List[OCRResult]:
        """Parse DeepSeek API response to OCRResult format"""
        results = []
        
        if "text_regions" in ocr_data:
            for region in ocr_data["text_regions"]:
                text = region.get("text", "").strip()
                if not text:
                    continue
                
                bbox_data = region.get("bbox", {})
                # Normalize bbox to image coordinates if needed
                bbox = BoundingBox(
                    x1=float(bbox_data.get("x1", 0)),
                    y1=float(bbox_data.get("y1", 0)),
                    x2=float(bbox_data.get("x2", image_size[0])),
                    y2=float(bbox_data.get("y2", image_size[1]))
                )
                
                results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    model_name=self.model_name,
                    confidence=region.get("confidence", 0.9)
                ))
        
        return results
