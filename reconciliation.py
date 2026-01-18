"""
LLM-based Voting / Reconciliation - Uses LLM to choose final text for each region
"""
from typing import List, Optional
import json

from models import TextCandidate, ReconciledResult, BoundingBox
from alignment import TextAlignment


class LLMReconciler:
    """Uses LLM to reconcile OCR results from multiple models"""
    
    def __init__(self, llm_client=None, model_name: str = "gpt-4"):
        """
        Initialize LLM reconciler
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.). If None, will try to auto-detect.
            model_name: Model name to use (e.g., "gpt-4", "claude-3-opus")
        """
        self.model_name = model_name
        self.llm_client = llm_client or self._get_default_client()
        self.aligner = TextAlignment()  # Reusable aligner instance
    
    def _get_default_client(self):
        """Try to get default LLM client (OpenAI by default)"""
        try:
            from openai import OpenAI
            # Will use OPENAI_API_KEY from environment
            return OpenAI()
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai\n"
                "Or provide a custom llm_client"
            )
    
    def reconcile(self, text_candidates: List[TextCandidate]) -> List[ReconciledResult]:
        """
        Reconcile multiple text candidates using LLM
        
        Args:
            text_candidates: List of TextCandidate objects, one per region
            
        Returns:
            List of ReconciledResult objects with final chosen text
        """
        reconciled_results = []
        
        for candidate in text_candidates:
            result = self._reconcile_region(candidate)
            if result:
                reconciled_results.append(result)
        
        return reconciled_results
    
    def _reconcile_region(self, text_candidate: TextCandidate) -> Optional[ReconciledResult]:
        """
        Reconcile candidates for a single region
        
        Returns ReconciledResult with final text choice
        """
        # Get unique candidates with their metadata
        unique_candidates = self.aligner.get_unique_candidates(text_candidate)
        
        if not unique_candidates:
            return None
        
        # If only one unique candidate, return it directly
        if len(unique_candidates) == 1:
            text, models, conf = unique_candidates[0]
            return ReconciledResult(
                text=text,
                bbox=text_candidate.region_bbox,
                confidence=conf,
                reason="Only one candidate available",
                source_models=models
            )
        
        # Build prompt for LLM
        prompt = self._build_reconciliation_prompt(unique_candidates)
        
        try:
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse response
            parsed = self._parse_llm_response(response)
            
            return ReconciledResult(
                text=parsed["text"],
                bbox=text_candidate.region_bbox,
                confidence=parsed.get("confidence", 0.8),
                reason=parsed.get("reason"),
                source_models=parsed.get("source_models", [])
            )
        except Exception as e:
            print(f"Error in LLM reconciliation: {e}")
            # Fallback: return highest confidence candidate
            text, models, conf = unique_candidates[0]
            return ReconciledResult(
                text=text,
                bbox=text_candidate.region_bbox,
                confidence=conf,
                reason=f"LLM reconciliation failed, using highest confidence: {str(e)}",
                source_models=models
            )
    
    def _build_reconciliation_prompt(self, candidates: List[tuple]) -> str:
        """Build prompt for LLM to reconcile candidates"""
        prompt = """You are an OCR quality assessment expert. Multiple OCR models have processed the same text region and produced different outputs.

Your task is to choose the best text from the candidates below. Consider:
1. Text accuracy and correctness
2. Model confidence scores
3. Common OCR errors (character substitutions, missing spaces, etc.)

Candidates:
"""
        for i, (text, models, conf) in enumerate(candidates, 1):
            prompt += f"{i}. Text: \"{text}\"\n"
            prompt += f"   Models: {', '.join(models)}\n"
            prompt += f"   Average Confidence: {conf:.2f}\n\n"
        
        prompt += """Please respond with a JSON object in this exact format:
{
  "text": "the best text choice",
  "confidence": 0.95,
  "reason": "brief explanation of why this text was chosen",
  "source_models": ["model1", "model2"]
}

Only return the JSON, no other text."""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt and return response"""
        if hasattr(self.llm_client, 'chat') or hasattr(self.llm_client, 'completions'):
            # OpenAI-style API
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"} if self.model_name.startswith("gpt-4") else None
                )
                return response.choices[0].message.content
            except AttributeError:
                # Fallback for older OpenAI API
                response = self.llm_client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=0.1
                )
                return response.choices[0].text
        
        # Generic callable
        elif callable(self.llm_client):
            return self.llm_client(prompt)
        
        else:
            raise ValueError(f"Unsupported llm_client type: {type(self.llm_client)}")
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM JSON response"""
        # Clean response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")


class SimpleVotingReconciler:
    """
    Simple voting-based reconciler (no LLM required)
    Uses majority voting and confidence weighting as fallback
    """
    
    def __init__(self):
        self.aligner = TextAlignment()
    
    def reconcile(self, text_candidates: List[TextCandidate]) -> List[ReconciledResult]:
        """Reconcile using simple voting"""
        reconciled_results = []
        
        for candidate in text_candidates:
            unique_candidates = self.aligner.get_unique_candidates(candidate)
            
            if not unique_candidates:
                continue
            
            # Use highest confidence candidate
            text, models, conf = unique_candidates[0]
            
            reconciled_results.append(ReconciledResult(
                text=text,
                bbox=candidate.region_bbox,
                confidence=conf,
                reason=f"Highest confidence candidate from {len(unique_candidates)} options",
                source_models=models
            ))
        
        return reconciled_results
