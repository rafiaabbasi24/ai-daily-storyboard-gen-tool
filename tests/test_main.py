import base64
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI application and the generation function.
# Adjust the import paths if the project structure differs.
from main import app
from generation import generate_storyboard

client = TestClient(app)


def test_health_endpoint():
    """
    Verify that the health check endpoint returns a 200 status
    and the expected JSON payload.
    """
    response = client.get("/health")
    assert response.status_code == 200, "Health endpoint should return HTTP 200"
    assert response.json() == {"status": "ok"}, "Health payload mismatch"


def test_generate_endpoint_success():
    """
    Test the /generate endpoint with a valid prompt.
    The underlying generation function is patched to avoid heavy model inference.
    """
    mock_storyboard = ["frame1_base64", "frame2_base64", "frame3_base64"]
    with patch("generation.generate_storyboard", return_value=mock_storyboard) as mock_gen:
        payload = {"prompt": "A sunny beach with palm trees"}
        response = client.post("/generate", json=payload)

        assert response.status_code == 200, "Successful generation should return HTTP 200"
        data = response.json()
        assert "storyboard" in data, "Response must contain a 'storyboard' key"
        assert data["storyboard"] == mock_storyboard, "Storyboard content mismatch"

        # Ensure the generation function was called with the correct arguments.
        mock_gen.assert_called_once_with("A sunny beach with palm trees", num_frames=5)


def test_generate_endpoint_missing_prompt():
    """
    The endpoint should reject requests that do not include a prompt.
    FastAPI automatically returns a 422 Unprocessable Entity for validation errors.
    """
    response = client.post("/generate", json={})
    assert response.status_code == 422, "Missing prompt should trigger validation error"


def test_generate_endpoint_invalid_json():
    """
    Sending malformed JSON should also result in a 422 error.
    """
    response = client.post(
        "/generate",
        data="not-a-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, "Invalid JSON payload should trigger validation error"


def test_generate_storyboard_logic():
    """
    Unit test for the core generation logic.
    The low‑level model call is patched so the test runs quickly and deterministically.
    """
    # Assume the internal helper that talks to the diffusion model is called
    # `_run_model` and returns raw image bytes.
    with patch("generation._run_model", return_value=b"dummy_image_bytes") as mock_run:
        num_frames = 2
        result = generate_storyboard(prompt="Test prompt", num_frames=num_frames)

        # The function should return a list of base64‑encoded strings.
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == num_frames, f"Result should contain {num_frames} frames"

        for item in result:
            # Each item must be a string that can be decoded from base64.
            assert isinstance(item, str), "Each storyboard frame should be a string"
            try:
                base64.b64decode(item, validate=True)
            except (base64.binascii.Error, ValueError) as exc:
                pytest.fail(f"Storyboard frame is not valid base64: {exc}")

        # Verify that the low‑level model helper was invoked once per frame.
        assert mock_run.call_count == num_frames, "Model should be called once per frame"


def test_generate_storyboard_custom_num_frames():
    """
    Ensure that the `num_frames` argument correctly influences the output length.
    """
    with patch("generation._run_model", return_value=b"img"):
        for frames in (1, 3, 5):
            result = generate_storyboard(prompt="Another test", num_frames=frames)
            assert len(result) == frames, f"Expected {frames} frames, got {len(result)}"


def test_generate_storyboard_prompt_passthrough():
    """
    Confirm that the prompt is correctly forwarded to the low‑level model call.
    """
    with patch("generation._run_model") as mock_run:
        mock_run.return_value = b"img"
        _ = generate_storyboard(prompt="Specific prompt", num_frames=2)
        # The mock should have been called twice with the same prompt.
        expected_calls = [(( "Specific prompt",),), (( "Specific prompt",),)]
        actual_calls = mock_run.call_args_list
        assert len(actual_calls) == 2, "Model should be invoked twice"
        for call in actual_calls:
            args, _ = call
            assert args[0] == "Specific prompt", "Prompt was not passed correctly to the model"
