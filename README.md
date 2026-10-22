# Conversational Storyboard Generator

A web application that turns natural language conversations into visual storyboards.  
The backend leverages **Diffusers** and **Transformers** to generate images from text prompts, while the frontend provides an interactive chat‑driven UI. The service is containerized with Docker and exposed via **FastAPI**.

---

## Table of Contents

- [Features](#features)  
- [Architecture](#architecture)  
- [Prerequisites](#prerequisites)  
- [Quick Start (Docker)](#quick-start-docker)  
- [Local Development](#local-development)  
  - [Backend](#backend)  
  - [Frontend](#frontend)  
- [API Reference](#api-reference)  
- [Environment Variables](#environment-variables)  
- [Testing](#testing)  
- [License](#license)  

---

## Features

- **Conversational UI** – type a story description and receive a sequence of generated frames.  
- **Stable Diffusion integration** via `diffusers` for high‑quality image synthesis.  
- **Prompt engineering** using `transformers` (e.g., GPT‑2) to expand user input into detailed frame prompts.  
- **Real‑time streaming** of generated images through FastAPI WebSocket endpoints.  
- **Docker‑first** – both backend and frontend run in isolated containers.  

---

## Architecture

```
┌─────────────────────┐      ┌─────────────────────┐
│   React Frontend    │ <--► │   FastAPI Backend   │
│ (Vite + TSX)        │      │ (Python 3.11)       │
└─────────────────────┘      └─────────────────────┘
          ▲                           ▲
          │ Docker Compose            │ Docker Compose
          ▼                           ▼
   ┌─────────────────┐          ┌─────────────────┐
   │  nginx (optional)│          │  GPU-enabled    │
   │  reverse proxy   │          │  container with│
   └─────────────────┘          │  diffusers      │
                                 └─────────────────┘
```

- **Backend** (`/backend`): FastAPI app exposing REST and WebSocket endpoints, handling prompt generation and image synthesis.  
- **Frontend** (`/frontend`): React app built with Vite, communicating with the backend via HTTP/WebSocket.  
- **Docker**: `docker-compose.yml` orchestrates three services – `backend`, `frontend`, and optional `nginx`.  

---

## Prerequisites

| Tool | Minimum Version |
|------|-----------------|
| Docker Engine | 24.0+ |
| Docker Compose | 2.20+ |
| NVIDIA driver (GPU builds only) | 525+ |
| Python | 3.11 (for local dev) |
| Node.js | 20.x (for local dev) |

> **Note:** GPU acceleration is optional. The CPU‑only image works out of the box but will be slower.

---

## Quick Start (Docker)

```bash
# Clone the repository
git clone https://github.com/your-org/conversational-storyboard-generator.git
cd conversational-storyboard-generator

# Build and start all services
docker compose up --build -d

# Wait for the backend to download model weights (first run only)
docker logs -f backend

# Open the UI
open http://localhost:3000   # macOS
# or navigate to http://localhost:3000 in your browser
```

The backend API is reachable at `http://localhost:8000/docs` (Swagger UI).

### Stop & Clean

```bash
docker compose down --remove-orphans
docker volume prune -f   # removes cached model weights if desired
```

---

## Local Development

### Backend

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# (Optional) Install pre‑commit hooks
pre-commit install

# Run the FastAPI server (auto‑reload)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### Running Tests

```bash
pytest backend/tests
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # starts Vite dev server on http://localhost:3000
```

The dev server proxies API calls to `http://localhost:8000` via Vite’s `proxy` config.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/generate` | Accepts a JSON payload `{ "story": "..." }`. Returns a list of frame IDs and URLs. |
| `GET`  | `/api/v1/frame/{frame_id}` | Streams the generated image (binary `image/png`). |
| `WS`   | `/ws/generate` | WebSocket for progressive image delivery. Sends JSON messages `{ "frame": 1, "url": "...", "status": "completed" }`. |

Full OpenAPI schema is available at `http://localhost:8000/docs`.

---

## Environment Variables

Create a `.env` file in the project root (or copy from `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `runwayml/stable-diffusion-v1-5` | Diffusers model identifier. |
| `DEVICE` | `cpu` | `cpu` or `cuda`. |
| `MAX_FRAMES` | `10` | Upper bound for storyboard length. |
| `PROMPT_MODEL` | `gpt2` | Transformers model for prompt expansion. |
| `CORS_ORIGINS` | `*` | Allowed origins for the FastAPI CORS middleware. |
| `LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warning`, `error`). |

---

## Testing

The backend includes unit and integration tests under `backend/tests`. Tests cover:

- Prompt expansion (`backend/services/prompt.py`)  
- Image generation pipeline (`backend/services/generator.py`) – mocked for CI  
- API endpoint validation (`backend/tests/test_api.py`)  

Run the full suite with:

```bash
pytest
```

The frontend uses **Vitest** and **React Testing Library**. To execute:

```bash
cd frontend
npm run test
```

---

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.