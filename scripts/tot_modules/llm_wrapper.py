#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Wrapper for LLaMA 3.2 and other models
"""

from typing import List, Dict, Optional
import torch
import requests
from transformers import pipeline
import warnings
import logging

# Suppress tokenizer warnings about max_length vs max_new_tokens
warnings.filterwarnings("ignore", message=".*max_length.*max_new_tokens.*")
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


class Llama32Chat:
    """
    Wrapper around Hugging Face transformers pipeline for LLaMA-3.2-3B-Instruct.
    Provides a chat-style interface for easy interaction.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
    ):
        """
        Initialize the LLM pipeline.
        Args:
            model_id: HuggingFace model identifier
            device_map: Device placement strategy ("auto", "cuda", "cpu")
            torch_dtype: Data type for model weights
        """
        self.model_id = model_id
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.tokenizer = self.pipe.tokenizer
        # Set pad_token for batching support
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Clear max_length to avoid conflicts with max_new_tokens
        self.tokenizer.model_max_length = 2147483647  # Use a very large value instead of model default

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        n: int = 1,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        """
        Generate responses using chat template.
        Args:
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0.0 = greedy)
            max_new_tokens: Maximum tokens to generate
            n: Number of completions to generate
        Returns:
            List of generated text strings
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        resolved_do_sample = (temperature > 0.0) if do_sample is None else do_sample

        if n > 1 and resolved_do_sample:
            # Batch: pass n copies of the prompt for parallel generation
            batch_prompts = [prompt] * n
            batch_out = self.pipe(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                batch_size=n,
            )
            outputs = [item[0]["generated_text"].strip() for item in batch_out]
        else:
            outputs = []
            for _ in range(n):
                out = self.pipe(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    do_sample=resolved_do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,
                )
                text = out[0]["generated_text"]
                outputs.append(text.strip())
        return outputs
    def __repr__(self) -> str:
        return f"Llama32Chat(model={self.model_id})"

class Qwen3CoderChat:
    """
    Wrapper around Hugging Face transformers pipeline for Qwen3-coder:30b.
    Provides a chat-style interface for easy interaction.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-coder-30B",
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
    ):
        self.model_id = model_id
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.tokenizer = self.pipe.tokenizer
        # Set pad_token for batching support
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Clear max_length to avoid conflicts with max_new_tokens
        self.tokenizer.model_max_length = 2147483647  # Use a very large value instead of model default

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        n: int = 1,
        do_sample: Optional[bool] = None,
        enable_thinking: bool = False,
    ) -> List[str]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        resolved_do_sample = (temperature > 0.0) if do_sample is None else do_sample

        if n > 1 and resolved_do_sample:
            # Batch: pass n copies of the prompt for parallel generation
            batch_prompts = [prompt] * n
            batch_out = self.pipe(
                batch_prompts,
        # Always use sequential generation to ensure temperature variation
        # (batch mode with identical prompts doesn't produce variation)
        outputs = []
        for _ in range(n):
            out = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
            )
            outputs = [item[0]["generated_text"].strip() for item in batch_out]
        else:
            outputs = []
            for _ in range(n):
                out = self.pipe(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    do_sample=resolved_do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,
                )
                text = out[0]["generated_text"]
                outputs.append(text.strip())
            text = out[0]["generated_text"]
            outputs.append(text.strip())
        return outputs

    def __repr__(self) -> str:
        return f"Qwen3CoderChat(model={self.model_id})"


class OllamaChat:
    """
    Wrapper for using Ollama API as a chat LLM backend.
    """
    def __init__(self, model_id: str = "qwen:latest", base_url: str = "http://localhost:11434"):
        self.model_id = model_id.replace("ollama:", "")
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        n: int = 1,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        # Ollama expects a single prompt string, so we concatenate messages
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        results = []
        for _ in range(n):
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_new_tokens
                }
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            # Robustly handle multi-line/streamed JSON responses
            try:
                # Try standard JSON first
                data = response.json()
                results.append(data.get("response", "").strip())
            except (requests.exceptions.JSONDecodeError, ValueError, Exception):
                # Fallback: parse each line as JSON and concatenate 'response' fields
                import json
                lines = response.text.strip().splitlines()
                response_text = ""
                for line in lines:
                    try:
                        obj = json.loads(line)
                        response_text += obj.get("response", "")
                    except Exception:
                        continue
                results.append(response_text.strip())
        return results

    def __repr__(self) -> str:
        return f"OllamaChat(model={self.model_id}, base_url={self.base_url})"
