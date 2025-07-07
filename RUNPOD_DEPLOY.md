# Deploy Stable Diffusion API on RunPod

## Method 1: Using RunPod Template (Recommended)

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Optimize for RunPod deployment"
   git push origin main
   ```

2. **Create a RunPod Template**:
   - Go to [RunPod](https://runpod.io)
   - Navigate to "Templates" → "New Template"
   - Fill in the details:
     - **Template Name**: `stable-diffusion-api`
     - **Container Image**: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`
     - **Docker Command**: 
       ```bash
       bash -c "cd /workspace && git clone https://github.com/NguyenDucAnforwork/sd-api.git && cd sd-api && pip install -r requirements.txt && python app.py"
       ```
     - **Expose HTTP Port**: `8000`
     - **Container Disk**: `20 GB`

3. **Deploy the Pod**:
   - Go to "Pods" → "Deploy"
   - Select your template
   - Choose a GPU (RTX 4090, A100, etc.)
   - Click "Deploy"

## Method 2: Using Docker Image

1. **Build and push Docker image**:
   ```bash
   # Build the image
   docker build -t your-dockerhub-username/sd-api:latest .
   
   # Push to Docker Hub
   docker push your-dockerhub-username/sd-api:latest
   ```

2. **Deploy on RunPod**:
   - Container Image: `your-dockerhub-username/sd-api:latest`
   - Expose HTTP Port: `8000`

## Method 3: Manual Setup

1. **Start a RunPod instance** with PyTorch template
2. **Open the terminal** and run:
   ```bash
   cd /workspace
   git clone https://github.com/NguyenDucAnforwork/sd-api.git
   cd sd-api
   pip install -r requirements.txt
   python app.py
   ```

## Accessing Your API

Once deployed, you can access your API at:
- **Health Check**: `https://your-pod-id-8000.proxy.runpod.net/health`
- **Generate Image**: `https://your-pod-id-8000.proxy.runpod.net/txt2img`

## Example Request

```bash
curl -X POST "https://your-pod-id-8000.proxy.runpod.net/txt2img" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful mountain landscape, photorealistic, 4k",
    "negative_prompt": "blurry, low quality",
    "height": 512,
    "width": 512,
    "num_inference_steps": 30,
    "guidance_scale": 7.5
  }'
```

## Tips for RunPod

- Choose GPU with at least 8GB VRAM (RTX 4090, A100)
- Monitor GPU usage via `/health` endpoint
- Use lower `num_inference_steps` for faster generation
- Enable spot instances for cost savings
