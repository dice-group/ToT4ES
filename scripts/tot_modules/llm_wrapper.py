#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Wrapper for LLaMA 3.2
"""

from typing import List, Dict
import torch
from transformers import pipeline


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

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        n: int = 1,
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
        # Convert chat messages into a single prompt using the model's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs: List[str] = []
        do_sample = temperature > 0.0

        for _ in range(n):
            out = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,  # only return the completion
            )
            # text-generation pipeline returns a list of dicts: [{'generated_text': '...'}]
            text = out[0]["generated_text"]
            outputs.append(text.strip())
        
        return outputs
    
    def __repr__(self) -> str:
        return f"Llama32Chat(model={self.model_id})"
