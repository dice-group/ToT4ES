#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Approach: Direct LLM Prompt for Entity Summarization

This baseline uses a direct instructional prompt to an LLM for entity summarization,
without any tree-of-thought reasoning. It serves as a comparison point for the ToT4ES approach.

The prompt follows the provided specification with strict selection rules:
1. Deduplication
2. Core facts prioritization
3. Metadata elimination
4. N-Triples output format
"""

import os
import sys
import logging
import warnings
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import torch
from transformers import pipeline

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
        """
        Normalize URI by ensuring it's wrapped in angle brackets.
        
        Args:
            uri: URI string (with or without brackets)
            
        Returns:
            URI wrapped in angle brackets
        """
        uri = uri.strip()
        if not uri.startswith('<'):
            uri = f"<{uri}"
        if not uri.endswith('>'):
            uri = f"{uri}>"
        return uri
    
    @staticmethod
    def parse_triple(line: str) -> Tuple[str, str, str]:
        """
        Parse a single N-Triples line into (subject, predicate, object).
        
        Args:
            line: N-Triples formatted line
            
        Returns:
            Tuple of (subject, predicate, object) or (None, None, None) if parse fails
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None
        
        # Remove trailing period if present
        if line.endswith(' .'):
            line = line[:-2].strip()
        elif line.endswith('.'):
            line = line[:-1].strip()
        
        # Basic parsing: split by whitespace but preserve URIs and literals
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
        # Normalize subject and predicate (wrap URIs in angle brackets)
        subject = NTriplesParser.normalize_uri(subject)
        predicate = NTriplesParser.normalize_uri(predicate)
        
        # Normalize object only if it's a URI (not a literal)
        if obj.startswith('<') or (not obj.startswith('"') and not obj.startswith("'")):
            # Check if it looks like a URI (doesn't start with quote)
            if not obj.startswith('"') and not obj.startswith("'"):
                obj = NTriplesParser.normalize_uri(obj)
        
        return f"{subject} {predicate} {obj} ."


class BaselineLLMSummarizer:
    """Direct LLM-based entity summarizer using instructional prompt."""
    
    # Metadata predicates to exclude
    METADATA_PREDICATES = {
        'thumbnail',
        'depiction',
        'wasDerivedFrom',
        'homepage',
        'hasPhotoCollection',
        'wikiPageWikiLink',
        'wikiPageExternalLink',
        'wikiPageID',
        'wikiPageRevisionID',
        'wikiPageLength',
        'abstract',  # Often too generic
    }
    
    # Predicate aliases for deduplication (maps to preferred form)
    PREDICATE_ALIASES = {
        'http://purl.org/dc/terms/subject': 'http://dbpedia.org/ontology/subject',
        'http://purl.org/dc/terms/knownFor': 'http://dbpedia.org/ontology/knownFor',
        'http://xmlns.com/foaf/0.1/name': 'http://www.w3.org/2000/01/rdf-schema#label',
    }
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
        model_local_dir: Optional[str] = None,
        download_model: bool = False,
    ):
        """
        Initialize the baseline summarizer with LLM.
        
        Args:
            model_id: HuggingFace model identifier
            device_map: Device placement strategy
            torch_dtype: Torch data type
        """
        model_source = self._resolve_model_source(
            model_id=model_id,
            model_local_dir=model_local_dir,
            download_model=download_model,
        )
        logger.info(f"Initializing Llama model from: {model_source}")
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_source,
                tokenizer=model_source,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            self.tokenizer = self.pipe.tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.model_max_length = 2147483647
            generation_config = self.pipe.model.generation_config
            generation_config.max_length = None
            generation_config.min_length = None
            # Some pipelines keep forward defaults with max_length/min_length.
            # Remove them so max_new_tokens/min_new_tokens are the only length controls.
            self.pipe._forward_params.pop("max_length", None)
            self.pipe._forward_params.pop("min_length", None)
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
        
        self.parser = NTriplesParser()

    @staticmethod
    def _resolve_model_source(
        model_id: str,
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
                f"Downloading model {model_id} to local directory: {local_path}"
            )
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                resume_download=True,
            )
            return str(local_path)

        model_path = Path(model_id).expanduser()
        if model_path.exists():
            return str(model_path.resolve())

        return model_id
    
    def load_triples(self, triple_file: str) -> List[str]:
        """
        Load raw triples from N-Triples file.
        
        Args:
            triple_file: Path to .nt file containing triples
            
        Returns:
            List of triple strings
        """
        triples = []
        try:
            with open(triple_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        triples.append(line)
            logger.info(f"Loaded {len(triples)} triples from {triple_file}")
        except Exception as e:
            logger.error(f"Error loading triples from {triple_file}: {e}")
            raise
        
        return triples
    
    def preprocess_triples(self, triples: List[str]) -> List[Tuple[str, str, str]]:
        """
        Parse the full raw triple set without method-specific filtering or deduplication.

        Args:
            triples: List of N-Triples strings

        Returns:
            List of valid (subject, predicate, object) tuples
        """
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
    
    def _is_metadata_predicate(self, predicate_name: str) -> bool:
        """Check if predicate is metadata that should be excluded."""
        for meta_pred in self.METADATA_PREDICATES:
            if meta_pred in predicate_name.lower():
                return True
        return False
    
    def format_full_triples_for_prompt(self, triples: List[Tuple[str, str, str]]) -> str:
        """
        Format triples as indexed full N-Triples to match ToT candidate representation.

        Args:
            triples: List of (subject, predicate, object) tuples

        Returns:
            Numbered N-Triples string
        """
        formatted = []
        for i, (s, p, o) in enumerate(triples, 1):
            formatted.append(f"{i}. {self.parser.format_triple(s, p, o)}")
        return '\n'.join(formatted)
    
    def create_prompt(
        self,
        entity_uri: str,
        entity_label: str,
        triples: List[Tuple[str, str, str]],
        summary_size: int = 5,
        prompt_style: str = "tot_matched",
    ) -> str:
        """
        Create the instructional prompt for LLM summarization.
        
        Args:
            entity_uri: URI of the entity
            entity_label: Human-readable label of entity
            triples: List of (subject, predicate, object) tuples
            summary_size: Target number of triples in summary
            
        Returns:
            Prompt string
        """
        formatted_triples = self.format_full_triples_for_prompt(triples)
        prompt = f"""You are an expert in entity summarization over RDF triples.

Entity: {entity_label}
Entity URI: {entity_uri}

You are given candidate triples for this entity. Select exactly {summary_size} triples that best summarize the entity.

Selection criteria:
1. Relatedness: choose triples central to the entity identity and core facts.
2. Informativeness: choose specific, non-trivial facts with strong information value.
3. Coverage/Diversity: cover different aspects of the entity and avoid redundant predicates.

Candidate triples (index: triple):
{formatted_triples}

Output rules:
- Output ONLY selected triples in valid N-Triples format, one per line.
- Select exactly {summary_size} triples from the candidates.
- Use {entity_uri} as subject for all output triples.
- All URIs must be wrapped in angle brackets.
- Each output line must end with " .".
- Do not output explanations, numbering, or extra text.
"""
        return prompt
    
    def summarize(
        self,
        entity_uri: str,
        entity_label: str,
        raw_triples: List[str],
        summary_size: int = 5,
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        prompt_style: str = "tot_matched",
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        """
        Summarize entity triples using direct LLM prompt.
        
        Args:
            entity_uri: URI of the entity
            entity_label: Human-readable entity label
            raw_triples: List of raw N-Triples strings
            summary_size: Target number of triples in summary
            temperature: LLM temperature
            max_new_tokens: Max tokens to generate
            
        Returns:
            List of selected triples in N-Triples format
        """
        logger.info(f"Summarizing entity: {entity_label}")
        logger.info(f"Input: {len(raw_triples)} triples, Target: {summary_size} triples")
        
        # Preprocess: filter and deduplicate
        processed_triples = self.preprocess_triples(raw_triples)
        
        if len(processed_triples) <= summary_size:
            logger.info(f"Input has {len(processed_triples)} unique triples, less than target {summary_size}")
            return [
                self.parser.format_triple(s, p, o)
                for s, p, o in processed_triples
            ]
        
        # Create prompt
        prompt = self.create_prompt(
            entity_uri,
            entity_label,
            processed_triples,
            summary_size,
            prompt_style,
        )
        
        logger.info("Sending prompt to LLM...")
        
        # Call LLM
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            prompt_template = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            resolved_do_sample = (temperature > 0.0) if do_sample is None else do_sample
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "temperature": temperature,
                "do_sample": resolved_do_sample,
                "max_length": None,
                "min_length": None,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
            }
            if top_p is not None and resolved_do_sample:
                generation_kwargs["top_p"] = top_p
            
            outputs = self.pipe(prompt_template, **generation_kwargs)
            
            response = outputs[0]['generated_text']
            
            # Extract the assistant's response (everything after the last "assistant" tag)
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            
            logger.info(f"LLM Response length: {len(response)} characters")
            
            # Parse response - expect N-Triples format
            # Pass entity_uri to ensure correct subject for all triples
            summary_triples = self._parse_llm_response(response, entity_uri)
            
            logger.info(f"Extracted {len(summary_triples)} triples from LLM response")
            
            return summary_triples
        
        except Exception as e:
            logger.error(f"Error during LLM summarization: {e}")
            raise
    
    def save_summary(
        self,
        summary: List[str],
        entity_id: int,
        summary_size: int,
        output_dir: str = "baseline_outputs",
        dataset_name: str = "dbpedia",
    ) -> str:
        """
        Save summary to .nt file in structured directory.
        
        Creates directory structure like:
        baseline_outputs/
        └── dbpedia_data/
            └── {entity_id}/
                └── {entity_id}_top{size}.nt
        
        Args:
            summary: List of N-Triple strings
            entity_id: Entity ID number
            summary_size: Summary size (e.g., 5, 10)
            output_dir: Base output directory (default: baseline_outputs)
            dataset_name: Dataset name (default: dbpedia, can be lmdb, faces)
            
        Returns:
            Path to saved file
        """
        # Create directory structure
        if dataset_name == "dbpedia":
            subdir = "dbpedia"
        elif dataset_name == "lmdb":
            subdir = "lmdb"
        elif dataset_name == "faces":
            subdir = "faces"
        else:
            subdir = dataset_name
        
        output_path = Path(output_dir) / subdir / str(entity_id)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{entity_id}_top{summary_size}.nt"
        filepath = output_path / filename
        
        # Write file with only N-Triples (no comments)
        with open(filepath, 'w', encoding='utf-8') as f:
            for triple in summary:
                f.write(triple + "\n")
        
        logger.info(f"Summary saved to {filepath}")
        return str(filepath)
    
    def save_summary_with_metadata(
        self,
        summary: List[str],
        entity_id: int,
        entity_uri: str,
        entity_label: str,
        summary_size: int,
        output_dir: str = "baseline_outputs",
        dataset_name: str = "dbpedia",
    ) -> str:
        """
        Save summary with detailed metadata.
        
        Args:
            summary: List of N-Triple strings
            entity_id: Entity ID number
            entity_uri: Entity URI
            entity_label: Entity label
            summary_size: Summary size
            output_dir: Base output directory
            dataset_name: Dataset name
            
        Returns:
            Path to saved file
        """
        # Create directory structure
        # Note: For OUTPUT, we use short names (dbpedia, lmdb, faces)
        # For INPUT, datasets use *_data (dbpedia_data, lmdb_data, faces_data)
        if dataset_name == "dbpedia":
            subdir = "dbpedia"
        elif dataset_name == "lmdb":
            subdir = "lmdb"
        elif dataset_name == "faces":
            subdir = "faces"
        else:
            subdir = dataset_name
        
        output_path = Path(output_dir) / subdir / str(entity_id)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{entity_id}_top{summary_size}.nt"
        filepath = output_path / filename
        
        # Write file with only N-Triples (no comments)
        with open(filepath, 'w', encoding='utf-8') as f:
            for triple in summary:
                f.write(triple + "\n")
        
        logger.info(f"Summary saved to {filepath}")
        return str(filepath)
    
    def _parse_llm_response(self, response: str, entity_uri: str = None) -> List[str]:
        """
        Parse LLM response to extract N-Triples.
        
        Args:
            response: Raw LLM response
            entity_uri: Expected entity URI to use as subject (overrides LLM's subject)
            
        Returns:
            List of valid N-Triples strings
        """
        triples = []
        lines = response.strip().split('\n')
        
        # Normalize the entity URI
        if entity_uri:
            entity_uri = self.parser.normalize_uri(entity_uri)
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to parse as N-Triple
            s, p, o = self.parser.parse_triple(line)
            if s is not None and p is not None and o is not None:
                # If entity_uri is provided, use it as the subject
                if entity_uri:
                    s = entity_uri
                formatted = self.parser.format_triple(s, p, o)
                triples.append(formatted)
        
        return triples


def main():
    """Demo/test the baseline summarizer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Baseline Direct LLM Summarizer for Entity Triples"
    )
    parser.add_argument(
        "--triple-file",
        type=str,
        required=True,
        help="Path to .nt file containing raw triples"
    )
    parser.add_argument(
        "--entity-id",
        type=int,
        required=True,
        help="Entity ID number (used for output directory structure)"
    )
    parser.add_argument(
        "--entity-uri",
        type=str,
        required=True,
        help="URI of the entity to summarize"
    )
    parser.add_argument(
        "--entity-label",
        type=str,
        required=True,
        help="Human-readable label of the entity"
    )
    parser.add_argument(
        "--summary-size",
        type=int,
        default=5,
        help="Target number of triples in summary (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="LLM model identifier"
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
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature for generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate (default: 1024)"
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
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="tot_matched",
        choices=["tot_matched"],
        help="Prompt template style (ToT-aligned single-pass)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_outputs",
        help="Base output directory for structured output (default: baseline_outputs)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dbpedia",
        choices=["dbpedia", "lmdb", "faces"],
        help="Dataset name for directory structure (default: dbpedia)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Custom output file path (overrides --output-dir structure)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (default: 0). Set to -1 to use CPU only"
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES string (e.g., '0,1' for GPUs 0 and 1)"
    )
    
    args = parser.parse_args()
    
    # Set CUDA devices if specified
    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    elif args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Initialize summarizer
    summarizer = BaselineLLMSummarizer(
        model_id=args.model,
        model_local_dir=args.model_local_dir,
        download_model=args.download_model,
    )
    
    # Load raw triples
    raw_triples = summarizer.load_triples(args.triple_file)
    
    # Summarize
    summary = summarizer.summarize(
        entity_uri=args.entity_uri,
        entity_label=args.entity_label,
        raw_triples=raw_triples,
        summary_size=args.summary_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
        top_p=args.top_p,
        do_sample=None if not args.no_sample else False,
    )
    
    # Determine output location
    if args.output_file:
        # Custom output file
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        output_path = args.output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Baseline Summary for {args.entity_label}\n")
            f.write(f"# Entity URI: {args.entity_uri}\n")
            f.write(f"# Summary size: {len(summary)} / {args.summary_size}\n")
            f.write("#\n")
            for triple in summary:
                f.write(triple + "\n")
        
        logger.info(f"Summary written to {output_path}")
    else:
        # Structured output directory
        output_path = summarizer.save_summary_with_metadata(
            summary=summary,
            entity_id=args.entity_id,
            entity_uri=args.entity_uri,
            entity_label=args.entity_label,
            summary_size=args.summary_size,
            output_dir=args.output_dir,
            dataset_name=args.dataset,
        )
    
    # Print info
    print(f"\n✓ Summary saved to: {output_path}")
    print(f"✓ Triples: {len(summary)}/{args.summary_size}")


if __name__ == "__main__":
    main()
