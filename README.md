# Multi-Model OCR Ensemble with LLM Reconciliation

A sophisticated OCR application that runs multiple OCR/vision-language models on the same document, normalizes their outputs, and uses an LLM to reconcile, rank, or merge the results.

## Architecture

### 1. OCR Ensemble Layer
- Runs multiple OCR models in parallel on the same image
- Supports: Tesseract, PaddleOCR, Surya OCR, DeepSeek OCR
- Generic design allows easy addition of new models
- Normalizes all outputs into a common schema

### 2. Alignment & Token Grouping
- Clusters overlapping bounding boxes using IoU (Intersection over Union)
- Groups candidate texts that refer to the same region
- Normalizes text (Unicode, whitespace, punctuation)
- Produces "text candidates per region"

### 3. LLM-based Voting / Reconciliation
- Uses structured reasoning with LLM (default: GPT-4)
- For each region, provides all candidate texts and confidence scores
- LLM outputs: final chosen text, confidence, and reasoning
- Fallback to simple voting if LLM unavailable

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Tesseract OCR, you also need to install the Tesseract binary:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

3. For PaddleOCR, ensure you have the dependencies:
```bash
# PaddleOCR will download models on first run
```

4. For Surya OCR:
```bash
# Surya OCR will automatically download models on first run
# Models are cached locally after first download
```

5. For DeepSeek OCR (optional - requires API key):
```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
```

6. For LLM reconciliation, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Command Line

```bash
# Basic usage with LLM reconciliation
python main.py input.png

# Save results to JSON
python main.py input.png -o output/results.json

# Use simple voting instead of LLM
python main.py path/to/image.png --no-llm

# Use different LLM model
python main.py path/to/image.png --llm-model gpt-3.5-turbo

# Adjust IoU threshold for bounding box clustering
python main.py path/to/image.png --iou-threshold 0.5
```

### Python API

```python
from main import OCRApp
from PIL import Image

# Initialize app
app = OCRApp(use_llm=True, llm_model="gpt-4")

# Process image
results = app.process_image("path/to/image.png", output_path="output.json")

# Access results
for result in results:
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence}")
    print(f"Bounding box: {result.bbox.to_dict()}")
```

## Adding New OCR Models

To add a new OCR model, extend the `OCRBase` class:

```python
from ocr_base import OCRBase
from models import OCRResult, BoundingBox, ModelName

class MyCustomOCR(OCRBase):
    def __init__(self):
        super().__init__(ModelName.MY_CUSTOM)  # Add enum value first
        # Initialize your model here
    
    def process(self, image: Image.Image) -> List[OCRResult]:
        # Process image and return OCRResult objects
        results = []
        # ... your OCR logic ...
        results.append(OCRResult(
            text="extracted text",
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
            model_name=self.model_name,
            confidence=0.95
        ))
        return results
```

Then add it to the ensemble:
```python
app = OCRApp(...)
app.ensemble.add_model(MyCustomOCR())
```

## Output Format

The application outputs reconciled results in this format:

```json
{
  "results": [
    {
      "text": "extracted text",
      "bbox": {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 70.0},
      "confidence": 0.95,
      "reason": "LLM explanation of choice",
      "source_models": ["tesseract", "paddleocr"]
    }
  ],
  "total_regions": 1
}
```

## Configuration

### IoU Threshold
Controls how bounding boxes are clustered. Lower values (0.1-0.3) group more strictly, higher values (0.5-0.7) group more loosely. Default: 0.3

### LLM Models
Supported models depend on your LLM client. Default is GPT-4, but you can use:
- `gpt-4`, `gpt-3.5-turbo` (OpenAI)
- Custom clients can be passed to `OCRApp(llm_client=...)`

## Limitations & Future Improvements

- **Saura & DeepSeek OCR**: Placeholder implementations - requires API integration
- **LLM Costs**: Each text region requires an LLM call. Consider batching or caching
- **Performance**: Can be optimized with async/await for parallel processing
- **Error Handling**: Some models may fail silently - check logs

## License

MIT
