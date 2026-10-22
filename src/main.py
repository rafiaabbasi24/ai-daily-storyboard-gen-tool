import os
import logging
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline
from diffusers import StableDiffusionPipeline
from io import BytesIO
from base64 import b64encode

# --------------------------------------------------------------------------- #
# Configuration & Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("storyboard")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt2")  # e.g., "gpt2", "facebook/opt-1.3b"
SD_MODEL = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #


class StoryboardRequest(BaseModel):
    """Request payload for storyboard generation."""

    prompt: str = Field(..., description="User supplied story prompt.")
    num_frames: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of storyboard frames to generate (1‑10).",
    )
    style: Optional[str] = Field(
        None,
        description="Optional textual style hint for image generation.",
    )


class Frame(BaseModel):
    """Single storyboard frame consisting of a description and an image."""

    text: str = Field(..., description="Generated description for the frame.")
    image_base64: str = Field(
        ...,
        description="Base64‑encoded PNG image representing the frame.",
    )


class StoryboardResponse(BaseModel):
    """Response containing all generated frames."""

    frames: List[Frame] = Field(..., description="List of storyboard frames.")


# --------------------------------------------------------------------------- #
# FastAPI app definition
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Conversational Storyboard Generator",
    description="Generate a textual storyboard and accompanying images from a prompt.",
    version="1.0.0",
)

# Allow the front‑end (React) to call the API from any origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Global pipeline objects – loaded once at startup
# --------------------------------------------------------------------------- #

llm: Optional[Pipeline] = None
sd_pipe: Optional[StableDiffusionPipeline] = None


def _load_llm() -> Pipeline:
    """Instantiate a text generation pipeline."""
    logger.info("Loading LLM model %s on %s", LLM_MODEL, DEVICE)
    return pipeline(
        "text-generation",
        model=LLM_MODEL,
        device=0 if DEVICE == "cuda" else -1,
        # Adjust generation parameters for more deterministic output
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
    )


def _load_sd() -> StableDiffusionPipeline:
    """Instantiate a Stable Diffusion pipeline."""
    logger.info("Loading Stable Diffusion model %s on %s", SD_MODEL, DEVICE)
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    pipe = pipe.to(DEVICE)
    # Enable attention slicing for lower memory usage on GPU
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
    return pipe


@app.on_event("startup")
def startup_event():
    """Load heavy models once when the server starts."""
    global llm, sd_pipe
    try:
        llm = _load_llm()
        sd_pipe = _load_sd()
        logger.info("Model loading complete.")
    except Exception as exc:
        logger.exception("Failed to load models during startup.")
        raise RuntimeError("Model initialization failed.") from exc


@app.on_event("shutdown")
def shutdown_event():
    """Release GPU memory on shutdown."""
    global llm, sd_pipe
    llm = None
    if sd_pipe is not None:
        sd_pipe = None
        torch.cuda.empty_cache()
    logger.info("Shutdown complete – resources released.")


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _generate_frame_text(base_prompt: str, frame_idx: int, total_frames: int) -> str:
    """
    Generate a concise description for a single storyboard frame.

    The LLM is prompted with the overall story and the index of the frame.
    """
    assert llm is not None, "LLM pipeline not initialized."
    prompt = (
        f"Story: {base_prompt}\n"
        f"Generate a short, vivid description for frame {frame_idx + 1} of {total_frames}."
    )
    logger.debug("LLM prompt for frame %d: %s", frame_idx + 1, prompt)
    result = llm(prompt, do_sample=True, max_new_tokens=60, temperature=0.8)
    # The pipeline returns a list of dicts with a 'generated_text' key.
    generated = result[0]["generated_text"]
    # Remove the original prompt part to keep only the new description.
    description = generated.replace(prompt, "").strip()
    # Ensure a single sentence.
    description = description.split("\n")[0].strip()
    logger.debug("Generated text for frame %d: %s", frame_idx + 1, description)
    return description


def _generate_image(description: str, style: Optional[str] = None) -> str:
    """
    Generate an image from a textual description using Stable Diffusion.

    Returns a base64‑encoded PNG.
    """
    assert sd_pipe is not None, "Stable Diffusion pipeline not initialized."
    prompt = description
    if style:
        prompt = f"{description}, in the style of {style}"
    logger.debug("SD prompt: %s", prompt)

    with torch.autocast("cuda" if DEVICE == "cuda" else "cpu"):
        image = sd_pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    b64_str = b64encode(img_bytes).decode("utf-8")
    logger.debug("Generated image size: %d bytes", len(img_bytes))
    return b64_str


# --------------------------------------------------------------------------- #
# API endpoint
# --------------------------------------------------------------------------- #


@app.post(
    "/generate",
    response_model=StoryboardResponse,
    summary="Generate a storyboard consisting of text and images.",
)
def generate_storyboard(request: StoryboardRequest) -> StoryboardResponse:
    """
    Orchestrates LLM and image generation to produce a multi‑frame storyboard.

    Steps:
    1. Use the LLM to create a short description for each frame.
    2. Feed each description (plus optional style) into Stable Diffusion.
    3. Return the frames with base64‑encoded PNG images.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    frames: List[Frame] = []
    for idx in range(request.num_frames):
        try:
            text = _generate_frame_text(
                base_prompt=request.prompt,
                frame_idx=idx,
                total_frames=request.num_frames,
            )
            image_b64 = _generate_image(description=text, style=request.style)
            frames.append(Frame(text=text, image_base64=image_b64))
        except Exception as exc:
            logger.exception("Failed to generate frame %d.", idx + 1)
            raise HTTPException(
                status_code=500,
                detail=f"Error generating frame {idx + 1}: {str(exc)}",
            ) from exc

    return StoryboardResponse(frames=frames)