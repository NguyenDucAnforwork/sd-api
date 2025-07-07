# Stable Diffusion 1.5 API

This project provides a FastAPI-based REST API for generating images using Stable Diffusion 1.5 with the diffusers library.

## Features

- Text-to-image generation with Stable Diffusion 1.5
- Configurable parameters (height, width, guidance scale, steps, etc.)
- Docker support for easy deployment

## Getting Started

### Prerequisites

- Docker
- Git

### Running with Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sd-api.git
   cd sd-api
   ```

2. Build the Docker image:
   ```bash
   docker build -t sd-api .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8000:8000 sd-api
   ```

4. The API will be available at `http://localhost:8000`

## API Usage

### Text to Image Generation

**Endpoint**: `/txt2img`

**Method**: POST

**Request Body**:
```json
{
  "prompt": "A beautiful mountain landscape",
  "negative_prompt": "blurry, bad quality",
  "height": 512,
  "width": 512,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42
}
```

**Response**:
```json
{
  "status": "success",
  "image": "base64_encoded_image_data",
  "parameters": {
    "prompt": "A beautiful mountain landscape",
    "negative_prompt": "blurry, bad quality",
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

### Health Check

**Endpoint**: `/health`

**Method**: GET

**Response**:
```json
{
  "status": "healthy"
}
```

## License

MIT
