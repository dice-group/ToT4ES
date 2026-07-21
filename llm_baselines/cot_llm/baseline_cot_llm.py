#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Approach: Chain-of-Thought (CoT) LLM for Entity Summarization

This baseline uses chain-of-thought prompting to encourage step-by-step reasoning
for entity summarization. The LLM is asked to:
1. Analyze the entity and understand its nature
2. Identify core facts and properties
3. Reason about importance and relevance
4. Select the top-k most important triples

This approach leverages the LLM's reasoning capabilities beyond direct prompting.
"""

import os
import sys
import logging
import warnings
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import torch
from transformers import pipeline
import argparse

# Suppress expected warnings from HuggingFace
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NTriplesParser:
    """Simple N-Triples parser for triple extraction and formatting."""
    
    @staticmethod
    def normalize_uri(uri: str) -> str:
        """Normalize URI by ensuring it's wrapped in angle brackets."""
        uri = uri.strip()
        if not uri.startswith('<'):
            uri = f"<{uri}"
        if not uri.endswith('>'):
            uri = f"{uri}>"
        return uri
    
    @staticmethod
    def parse_triple(line: str) -> Tuple[str, str, str]:
        """Parse a single N-Triples line into (subject, predicate, object)."""
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None
        
        # Remove trailing period if present
        if line.endswith(' .'):
            line = line[:-2].strip()
        elif line.endswith('.'):
            line = line[:-1].strip()
        
        # Basic parsing
        parts = []
        current = ""
        in_uri = False
        in_literal = False
        escape = False
        
        for char in line:
            if char == '<' and not in_literal:
                in_uri = True
                current += char
            elif char == '>' and in_uri:
                in_uri = False
                current += char
            elif char == '"' and not escape:
                in_literal = not in_literal
                current += char
            elif char == '\\' and in_literal:
                escape = True
                current += char
                continue
            elif char == ' ' and not in_uri and not in_literal:
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += char
            escape = False
        
        if current:
            parts.append(current)
        
        if len(parts) >= 3:
            return parts[0], parts[1], ' '.join(parts[2:])
        
        return None, None, None
    
    @staticmethod
    def format_triple(subject: str, predicate: str, obj: str) -> str:
        """Format a triple in N-Triples format with proper angle brackets."""
        subject = NTriplesParser.normalize_uri(subject)
        predicate = NTriplesParser.normalize_uri(predicate)
        
        if obj.startswith('<') or (not obj.startswith('"') and not obj.startswith("'")):
            if not obj.startswith('"') and not obj.startswith("'"):
                obj = NTriplesParser.normalize_uri(obj)
        
        return f"{subject} {predicate} {obj} ."


class CoTLLMSummarizer:
    """Chain-of-Thought LLM-based entity summarizer."""
    
    # Metadata predicates to exclude
    METADATA_PREDICATES = {
        'thumbnail', 'depiction', 'wasDerivedFrom', 'homepage',
        'hasPhotoCollection', 'wikiPageWikiLink', 'wikiPageExternalLink',
        'wikiPageID', 'wikiPageRevisionID', 'wikiPageLength', 'abstract',
    }
    
    PREDICATE_ALIASES = {
        # Common aliases
        '<http://dbpedia.org/ontology/name>': '<http://dbpedia.org/ontology/name>',
        '<http://xmlns.com/foaf/0.1/name>': '<http://dbpedia.org/ontology/name>',
    }
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: int = 0,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        model_local_dir: Optional[str] = None,
        download_model: bool = False,
    ):
        """
        Initialize CoT LLM summarizer.
        
        Args:
            model_name: HuggingFace model identifier
            device: GPU device ID
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens for generation
        """
        self.parser = NTriplesParser()
        model_source = self._resolve_model_source(
            model_name=model_name,
            model_local_dir=model_local_dir,
            download_model=download_model,
        )
        self.model_name = model_source
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Loading model: {model_source}")
        
        # Setup device
        if torch.cuda.is_available():
            # Check if CUDA_VISIBLE_DEVICES is set - if so, device becomes 0
            import os
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                actual_device = 0
                logger.info(f"CUDA_VISIBLE_DEVICES is set, using device 0 (mapped to GPU {device})")
            else:
                actual_device = device
                logger.info(f"Using GPU device: {device}")
            
            torch.cuda.set_device(actual_device)
        else:
            logger.info("GPU not available, using CPU")
        
        # Load pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model_source,
            tokenizer=model_source,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        generation_config = self.pipe.model.generation_config
        generation_config.max_length = None
        generation_config.min_length = None
        # Some pipelines keep forward defaults with max_length/min_length.
        # Remove them so max_new_tokens/min_new_tokens are the only length controls.
        self.pipe._forward_params.pop("max_length", None)
        self.pipe._forward_params.pop("min_length", None)

    @staticmethod
    def _resolve_model_source(
        model_name: str,
        model_local_dir: Optional[str],
        download_model: bool,
    ) -> str:
        """Resolve model source path, optionally downloading once to a local directory."""
        if model_local_dir:
            local_path = Path(model_local_dir).expanduser().resolve()
            if local_path.exists() and any(local_path.iterdir()):
                return str(local_path)

            if not download_model:
                raise FileNotFoundError(
                    f"Model local directory not found or empty: {local_path}. "
                    "Use --download-model to fetch model files once."
                )

            try:
                from huggingface_hub import snapshot_download
            except Exception as e:
                raise RuntimeError(
                    "huggingface_hub is required for --download-model. "
                    "Install it with: pip install huggingface_hub"
                ) from e

            logger.info(
                f"Downloading model {model_name} to local directory: {local_path}"
            )
            snapshot_download(
                repo_id=model_name,
                local_dir=str(local_path),
                resume_download=True,
            )
            return str(local_path)

        model_path = Path(model_name).expanduser()
        if model_path.exists():
            return str(model_path.resolve())

        return model_name
    
    def load_triples(self, triple_file: str) -> List[str]:
        """Load raw triples from N-Triples file."""
        triples = []
        try:
            with open(triple_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        triples.append(line)
            logger.info(f"Loaded {len(triples)} triples from {triple_file}")
        except Exception as e:
            logger.error(f"Error loading triples: {e}")
            raise
        
        return triples
    
    def _is_metadata_predicate(self, predicate_name: str) -> bool:
        """Check if predicate is metadata/non-content."""
        return predicate_name.lower() in self.METADATA_PREDICATES
    
    def preprocess_triples(self, triples: List[str]) -> List[Tuple[str, str, str]]:
        """Parse the full raw triple set without method-specific filtering or deduplication."""
        parsed_triples = []

        for triple_str in triples:
            s, p, o = self.parser.parse_triple(triple_str)
            if s is None:
                continue
            parsed_triples.append((s, p, o))

        logger.info(
            f"Prepared {len(parsed_triples)} valid triples without filtering or deduplication"
        )
        return parsed_triples
    
    def create_cot_prompt(
        self,
        entity_uri: str,
        entity_label: str,
        triples: List[Tuple[str, str, str]],
        summary_size: int = 5,
    ) -> str:
        """
        Create a CoT prompt aligned with the ToT4ES semantic criteria.
        """
        formatted_triples = "\n".join(
            [f"{i+1}. {self.parser.format_triple(s, p, o)}" for i, (s, p, o) in enumerate(triples)]
        )
        
        prompt = f"""You are an expert in entity summarization over RDF triples.

Entity: {entity_label}
Entity URI: {entity_uri}

You are given candidate triples for this entity. Select exactly {summary_size} triples that best summarize the entity.

Selection criteria (same semantic dimensions as ToT4ES):
1. Relatedness: choose triples central to the entity identity and core facts.
2. Informativeness: choose specific, non-trivial facts with strong information value.
3. Coverage/Diversity: cover different aspects of the entity and avoid redundant predicates.

Candidate triples (index: triple):
{formatted_triples}

Task:
- Think step-by-step about which triples best satisfy the three criteria jointly.
- You may reason briefly, but the final answer must contain only the selected triples.

Output rules:
- Select exactly {summary_size} triples from the candidates.
- Use {entity_uri} as subject for all output triples.
- All URIs must be wrapped in angle brackets.
- Each output line must end with " .".
- Do not use numbering or bullet points in the final answer.

After your reasoning, output this header exactly:
FINAL SELECTED TRIPLES:

Then output ONLY the selected triples in valid N-Triples format, one per line.
"""
        return prompt
    
    def _parse_cot_response(self, response: str, entity_uri: str) -> List[str]:
        """
        Parse CoT response to extract selected triples.
        
        The response contains reasoning and final selection.
        We extract the N-Triples lines from the final selection section.
        """
        lines = response.split('\n')
        selected_triples = []
        
        # Look for the final selection section and extract N-Triples lines
        in_selection = False
        for line in lines:
            original_line = line
            line = line.strip()
            
            # Track if we're in the final selection section
            if 'FINAL' in line.upper() or 'Top' in line:
                in_selection = True
                logger.debug(f"Found selection header: {line}")
                continue
            
            # Skip empty lines and obvious headers
            if not line:
                continue
            
            # Extract potentially valid N-Triples lines
            if in_selection:
                # Remove numbering/bullet points if present (e.g., "1. ", "- ", "* ")
                clean_line = line
                if clean_line and clean_line[0].isdigit() and '. ' in clean_line[:4]:
                    clean_line = clean_line.split('. ', 1)[1].strip()
                elif clean_line.startswith('-'):
                    clean_line = clean_line[1:].strip()
                elif clean_line.startswith('*'):
                    clean_line = clean_line[1:].strip()
                
                # Check if it's a valid N-Triples line
                # Must have at least 2 URIs (subject and predicate) and end with a period
                # Be flexible with spacing around the period
                if clean_line and '<' in clean_line:
                    # Count opening angle brackets
                    if clean_line.count('<') >= 2:
                        # Should end with a period (with or without spaces)
                        if clean_line.rstrip().endswith('.'):
                            logger.debug(f"Extracted triple: {clean_line}")
                            selected_triples.append(clean_line)
                        # Also try lines that look like N-Triples but might be missing period
                        elif '> ' in clean_line and clean_line.count('<') >= 2:
                            # Likely an N-Triple missing the final period, add it
                            if not clean_line.endswith(' .'):
                                clean_line = clean_line.rstrip() + ' .'
                            logger.debug(f"Extracted (added period): {clean_line}")
                            selected_triples.append(clean_line)
        
        logger.info(f"Extracted {len(selected_triples)} triples from response")
        
        # Ensure entity_uri is applied to all triples
        final_triples = []
        for triple in selected_triples:
            s, p, o = self.parser.parse_triple(triple)
            if s and p and o:
                # Apply the correct entity_uri as subject
                final_triple = self.parser.format_triple(entity_uri, p, o)
                final_triples.append(final_triple)
        
        return final_triples
    
    def summarize(
        self,
        entity_id: str,
        entity_uri: str,
        entity_label: str,
        triple_file: str,
        summary_size: int = 5,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        """
        Summarize an entity using CoT prompting.
        
        Args:
            entity_id: Entity identifier
            entity_uri: Full entity URI
            entity_label: Human-readable label
            triple_file: Path to N-Triples file
            summary_size: Number of triples to select
            
        Returns:
            List of selected N-Triples strings
        """
        logger.info(f"Summarizing entity {entity_id}: {entity_label}")
        
        # Load and preprocess triples
        raw_triples = self.load_triples(triple_file)
        processed_triples = self.preprocess_triples(raw_triples)
        
        if not processed_triples:
            logger.warning(f"No triples available for entity {entity_id}")
            return []
        
        logger.info(f"Processing {len(processed_triples)} triples for entity {entity_id}")
        
        # Create CoT prompt
        prompt = self.create_cot_prompt(
            entity_uri, entity_label, processed_triples, summary_size
        )
        
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Generate response using LLM
        logger.info(f"Generating summary for entity {entity_id}...")
        try:
            resolved_temperature = self.temperature if temperature is None else temperature
            resolved_max_new_tokens = self.max_tokens if max_new_tokens is None else max_new_tokens
            resolved_do_sample = (resolved_temperature > 0.0) if do_sample is None else do_sample

            generation_kwargs = {
                "max_new_tokens": resolved_max_new_tokens,
                "min_new_tokens": 1,
                "temperature": resolved_temperature,
                "do_sample": resolved_do_sample,
                "max_length": None,
                "min_length": None,
                "return_full_text": False,
                "pad_token_id": self.pipe.tokenizer.eos_token_id,
            }
            if top_p is not None and resolved_do_sample:
                generation_kwargs["top_p"] = top_p

            response = self.pipe(prompt, **generation_kwargs)
            
            response_text = response[0]['generated_text']
            logger.debug(f"Response length: {len(response_text)} characters")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return []
        
        # Parse response to extract triples
        selected_triples = self._parse_cot_response(response_text, entity_uri)
        
        if not selected_triples:
            logger.warning(f"No triples extracted from response for entity {entity_id}")
            logger.info(f"Falling back to first {summary_size} preprocessed triples")
            
            # Fallback: use the first summary_size triples from preprocessed
            fallback_triples = processed_triples[:summary_size]
            selected_triples = [
                self.parser.format_triple(entity_uri, p, o) 
                for _, p, o in fallback_triples
            ]
            
            if selected_triples:
                logger.info(f"Using fallback with {len(selected_triples)} triples")
            else:
                logger.error(f"Fallback also failed for entity {entity_id}")
                return []
        
        # Ensure we have the correct number of triples
        if len(selected_triples) < summary_size:
            logger.warning(f"Got {len(selected_triples)} triples, expected {summary_size}")
            # Pad with additional fallback triples if needed
            if len(selected_triples) < len(processed_triples):
                remaining = processed_triples[len(selected_triples):summary_size]
                for _, p, o in remaining:
                    selected_triples.append(self.parser.format_triple(entity_uri, p, o))
                logger.info(f"Padded with fallback to reach {len(selected_triples)} triples")
        elif len(selected_triples) > summary_size:
            logger.info(f"Truncating to {summary_size} triples (got {len(selected_triples)})")
            selected_triples = selected_triples[:summary_size]
        
        logger.info(f"Successfully generated summary with {len(selected_triples)} triples")
        return selected_triples


def main():
    parser = argparse.ArgumentParser(
        description="CoT-based Baseline LLM Entity Summarizer"
    )
    parser.add_argument(
        "--entity-id",
        type=str,
        required=True,
        help="Entity ID to summarize"
    )
    parser.add_argument(
        "--entity-uri",
        type=str,
        help="Entity URI (auto-discovered from _desc.nt if not provided)"
    )
    parser.add_argument(
        "--entity-label",
        type=str,
        default="Entity",
        help="Human-readable entity label"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input N-Triples file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_cot_outputs",
        help="Output directory for summaries"
    )
    parser.add_argument(
        "--summary-size",
        type=int,
        default=5,
        help="Summary size (number of triples to select)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model to use"
    )
    parser.add_argument(
        "--model-local-dir",
        type=str,
        default=None,
        help="Optional local directory containing model files"
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download model to --model-local-dir if not present"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top-p nucleus sampling value when sampling is enabled"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Force greedy decoding regardless of temperature"
    )
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = CoTLLMSummarizer(
        model_name=args.model,
        device=args.gpu,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        model_local_dir=args.model_local_dir,
        download_model=args.download_model,
    )
    
    # Auto-discover entity URI if not provided
    if not args.entity_uri:
        try:
            parser_obj = NTriplesParser()
            with open(args.input_file, 'r') as f:
                first_line = f.readline()
                uri, _, _ = parser_obj.parse_triple(first_line)
                if uri:
                    args.entity_uri = uri
                    logger.info(f"Discovered entity URI: {args.entity_uri}")
        except Exception as e:
            logger.warning(f"Could not auto-discover URI: {e}")
            args.entity_uri = f"<http://example.org/{args.entity_id}>"
    
    # Generate summary
    summary = summarizer.summarize(
        entity_id=args.entity_id,
        entity_uri=args.entity_uri,
        entity_label=args.entity_label,
        triple_file=args.input_file,
        summary_size=args.summary_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        do_sample=None if not args.no_sample else False,
    )
    
    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        args.entity_id,
        f"{args.entity_id}_top{args.summary_size}.nt"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for triple in summary:
            f.write(triple + '\n')
    
    logger.info(f"Saved summary to: {output_file}")
    print(f"\nSummary saved to: {output_file}")
    print(f"Summary size: {len(summary)} triples")


if __name__ == "__main__":
    main()
