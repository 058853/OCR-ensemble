"""
Main OCR Ensemble Application
Runs multiple OCR models, aligns outputs, and reconciles using LLM
"""
import argparse
from pathlib import Path
from PIL import Image
from typing import List, Optional

from ocr_models import TesseractOCR, PaddleOCRModel, SuryaOCR, DeepSeekOCR
from ensemble import OCREnsemble
from alignment import TextAlignment
from reconciliation import LLMReconciler, SimpleVotingReconciler
from models import ReconciledResult
import os
os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata/"


class OCRApp:
    """Main OCR application with ensemble and reconciliation"""
    
    def __init__(self, use_llm: bool = True, llm_model: str = "gpt-4", llm_client=None):
        """
        Initialize OCR application
        
        Args:
            use_llm: Whether to use LLM for reconciliation (False = simple voting)
            llm_model: LLM model name
            llm_client: Optional LLM client instance
        """
        # Initialize OCR models
        self.models = []
        
        # Add available models
        try:
            self.models.append(TesseractOCR())
            print("✓ Tesseract OCR initialized")
        except Exception as e:
            print(f"✗ Tesseract OCR not available: {e}")
        
        try:
            self.models.append(PaddleOCRModel())
            print("✓ PaddleOCR initialized")
        except Exception as e:
            print(f"✗ PaddleOCR not available: {e}")
        
        try:
            self.models.append(SuryaOCR())
            print("✓ Surya OCR initialized")
        except Exception as e:
            print(f"✗ Surya OCR not available: {e}")
        
        # # DeepSeek requires API key
        # try:
        #     deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        #     if deepseek_key:
        #         self.models.append(DeepSeekOCR(api_key=deepseek_key))
        #         print("✓ DeepSeek OCR initialized")
        #     else:
        #         print("⊘ DeepSeek OCR skipped (DEEPSEEK_API_KEY not set)")
        # except Exception as e:
        #     print(f"✗ DeepSeek OCR not available: {e}")
        
        if not self.models:
            raise RuntimeError("No OCR models available. Please install at least one OCR library.")
        
        # Initialize ensemble
        self.ensemble = OCREnsemble(self.models)
        
        # Initialize alignment
        self.alignment = TextAlignment(iou_threshold=0.3)
        
        # Initialize reconciliation
        if use_llm:
            try:
                self.reconciler = LLMReconciler(llm_client=llm_client, model_name=llm_model)
                print(f"✓ LLM Reconciler initialized (model: {llm_model})")
            except Exception as e:
                print(f"✗ LLM Reconciler not available: {e}. Falling back to simple voting.")
                self.reconciler = SimpleVotingReconciler()
        else:
            self.reconciler = SimpleVotingReconciler()
            print("✓ Using simple voting reconciler")
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> List[ReconciledResult]:
        """
        Process an image through the complete OCR pipeline
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save results (JSON format)
            
        Returns:
            List of ReconciledResult objects
        """
        # Load image
        image = Image.open(image_path)
        print(f"\nProcessing image: {image_path}")
        print(f"Image size: {image.size}")
        
        # Step 1: Run OCR ensemble
        print("\n[Step 1] Running OCR ensemble...")
        ocr_results = self.ensemble.process(image, parallel=True)
        print(f"  → Got {len(ocr_results)} results from {len(self.models)} models")
        for model in self.models:
            model_results = [r for r in ocr_results if r.model_name == model.model_name]
            print(f"    - {model.model_name}: {len(model_results)} results")
        
        # Step 2: Align and group by region
        print("\n[Step 2] Aligning and grouping by region...")
        text_candidates = self.alignment.group_by_region(ocr_results)
        print(f"  → Found {len(text_candidates)} distinct text regions")
        
        # Step 3: Reconcile using LLM or voting
        print("\n[Step 3] Reconciling results...")
        reconciled_results = self.reconciler.reconcile(text_candidates)
        print(f"  → Reconciled to {len(reconciled_results)} final results")
        
        # Save results if output path provided
        if output_path:
            self._save_results(reconciled_results, output_path)
            print(f"\nResults saved to: {output_path}")
        
        return reconciled_results
    
    def _save_results(self, results: List[ReconciledResult], output_path: str):
        """Save results to JSON file"""
        import json
        results_dict = {
            "results": [r.to_dict() for r in results],
            "total_regions": len(results)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    def print_results(self, results: List[ReconciledResult]):
        """Pretty print results to console"""
        print("\n" + "="*80)
        print("FINAL RECONCILED OCR RESULTS")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Region {i}")
            print(f"  Text: {result.text}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Bounding Box: ({result.bbox.x1:.1f}, {result.bbox.y1:.1f}) → ({result.bbox.x2:.1f}, {result.bbox.y2:.1f})")
            if result.source_models:
                print(f"  Source Models: {', '.join(result.source_models)}")
            if result.reason:
                print(f"  Reason: {result.reason}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Multi-model OCR Ensemble with LLM Reconciliation")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file path")
    parser.add_argument("--no-llm", action="store_true", help="Use simple voting instead of LLM")
    parser.add_argument("--llm-model", type=str, default="gpt-4", help="LLM model name (default: gpt-4)")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for bbox clustering (default: 0.3)")
    
    args = parser.parse_args()
    
    # Verify image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Initialize app
    app = OCRApp(use_llm=not args.no_llm, llm_model=args.llm_model)
    
    # Process image
    results = app.process_image(args.image, args.output)
    
    # Print results
    app.print_results(results)


if __name__ == "__main__":
    main()
