import os
import threading
from typing import Any, Dict, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Internal helpers – thread‑safe lazy loading of models
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_text_model: Optional[AutoModelForCausalLM] = None
_text_tokenizer: Optional[AutoTokenizer] = None
_image_pipe: Optional[StableDiffusionPipeline] = None


def _get_device() -> str:
    """
    Return the best available torch device.

    Returns
    -------
    str
        ``'cuda'`` if a GPU is available, otherwise ``'cpu'``.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_text_model(
    model_name: str = "Qwen/Qwen-7B-Chat",
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> None:
    """
    Load the Qwen text generation model and tokenizer lazily.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier for the Qwen model.
    device : str, optional
        Torch device to move the model onto. If ``None`` the function
        determines the best device automatically.
    torch_dtype : torch.dtype, optional
        Data type for model weights. ``float16`` is used for GPU
        inference to reduce memory consumption.
    """
    global _text_model, _text_tokenizer

    if device is None:
        device = _get_device()

    with _lock:
        if _text_model is None or _text_tokenizer is None:
            # Load tokenizer first – it does not depend on the device.
            _text_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True,
            )
            # Load the model with the appropriate dtype and device map.
            _text_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
            )
            _text_model.to(device)


def _load_image_pipeline(
    model_name: str = "stabilityai/stable-diffusion-2-1",
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> None:
    """
    Load the Stable Diffusion pipeline lazily.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier for the Stable Diffusion checkpoint.
    device : str, optional
        Torch device to move the pipeline onto. If ``None`` the function
        determines the best device automatically.
    torch_dtype : torch.dtype, optional
        Data type for model weights. ``float16`` is used for GPU inference.
    """
    global _image_pipe

    if device is None:
        device = _get_device()

    with _lock:
        if _image_pipe is None:
            _image_pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                revision="fp16" if device == "cuda" else "main",
                use_auth_token=os.getenv("HF_TOKEN", None),
            )
            _image_pipe = _image_pipe.to(device)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def generate_narration(
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_name: str = "Qwen/Qwen-7B-Chat",
    device: Optional[str] = None,
    **model_kwargs: Any,
) -> str:
    """
    Generate a story narration using the Qwen language model.

    Parameters
    ----------
    prompt : str
        The textual prompt that seeds the generation.
    max_new_tokens : int, optional
        Maximum number of tokens to generate. Defaults to ``256``.
    temperature : float, optional
        Sampling temperature. Higher values increase randomness.
    top_p : float, optional
        Nucleus sampling probability. ``0.9`` is a common default.
    model_name : str, optional
        HuggingFace identifier for the Qwen model.
    device : str, optional
        Torch device (``'cpu'`` or ``'cuda'``). If ``None`` the function
        auto‑detects the best device.
    **model_kwargs : Any
        Additional keyword arguments forwarded to ``model.generate``.

    Returns
    -------
    str
        The generated narration text, stripped of leading/trailing whitespace.
    """
    _load_text_model(model_name=model_name, device=device)

    assert _text_model is not None and _text_tokenizer is not None  # for mypy

    # Tokenize the prompt
    inputs = _text_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(_text_model.device)

    # Generate tokens
    output_ids = _text_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        **model_kwargs,
    )

    # Decode and post‑process
    narration = _text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return narration.strip()


def generate_image(
    prompt: str,
    *,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None,
    model_name: str = "stabilityai/stable-diffusion-2-1",
    device: Optional[str] = None,
    **pipeline_kwargs: Any,
) -> Image.Image:
    """
    Generate an image from a textual prompt using Stable Diffusion.

    Parameters
    ----------
    prompt : str
        Text prompt describing the desired image.
    num_inference_steps : int, optional
        Number of denoising steps. More steps usually improve quality.
    guidance_scale : float, optional
        Classifier‑free guidance weight. Higher values enforce prompt adherence.
    height : int, optional
        Height of the generated image in pixels. Must be a multiple of 8.
    width : int, optional
        Width of the generated image in pixels. Must be a multiple of 8.
    seed : int, optional
        Random seed for reproducibility. If ``None`` a random seed is used.
    model_name : str, optional
        HuggingFace identifier for the Stable Diffusion checkpoint.
    device : str, optional
        Torch device (``'cpu'`` or ``'cuda'``). Auto‑detected if ``None``.
    **pipeline_kwargs : Any
        Additional arguments passed to the pipeline's ``__call__`` method.

    Returns
    -------
    PIL.Image.Image
        The generated image.
    """
    _load_image_pipeline(model_name=model_name, device=device)

    assert _image_pipe is not None  # for mypy

    generator = torch.Generator(device=_image_pipe.device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    # The pipeline returns a dict with a key "images"
    result = _image_pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **pipeline_kwargs,
    )
    # ``result.images`` is a list; we return the first image.
    image = result.images[0]
    if not isinstance(image, Image.Image):
        raise TypeError("Stable Diffusion pipeline did not return a PIL.Image.")
    return image


def clear_caches() -> None:
    """
    Release model and pipeline objects from memory.

    This is useful in environments where the process may need to
    reload models with different configurations (e.g., during testing).
    """
    global _text_model, _text_tokenizer, _image_pipe
    with _lock:
        _text_model = None
        _text_tokenizer = None
        if _image_pipe is not None:
            _image_pipe.scheduler = None  # break possible circular refs
        _image_pipe = None
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Example usage (removed in production but kept for reference)
# --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     story = generate_narration("A brave knight enters a dark forest.")
#     print(story)
#     img = generate_image("A mystical forest illuminated by moonlight, digital art")
#     img.save("forest.png")