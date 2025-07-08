import io
import torch
import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import uvicorn

app = FastAPI(title="Stable Diffusion 1.5 API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the pipeline
pipe = None

# Initialize the model
@app.on_event("startup")
async def startup_event():
    global pipe
    print("Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Optimize for RunPod GPU environment
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        # Enable memory efficient attention for better GPU utilization
        pipe.enable_attention_slicing()
        pipe.enable_memory_efficient_attention()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        pipe = pipe.to("cpu")
        print("Warning: CUDA is not available. Running on CPU will be slow.")

class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = None

@app.post("/txt2img")
async def text_to_image(request: TextToImageRequest):
    try:
        print(f"Received request: {request.prompt[:50]}...")
        
        # Validate that model is loaded
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        # Clear GPU cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Set a seed if provided
        if request.seed is not None:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(request.seed)
        else:
            generator = None
            
        print(f"Generating image with prompt: {request.prompt[:50]}...")
        
        # Generate the image with progress callback
        def progress_callback(step, timestep, latents):
            print(f"Step {step}/{request.num_inference_steps}")
        
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            callback=progress_callback,
            callback_steps=5
        ).images[0]
        
        # Save image to generated_images folder
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        image_path = f"/app/generated_images/{filename}"
        image.save(image_path)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print("Image generated successfully!")
        print(f"Image size: {len(img_str)} characters")
        print(f"Image saved to: {image_path}")
        
        return {
            "status": "success",
            "image": img_str,
            "image_url": f"/images/{filename}",
            "image_info": {
                "format": "PNG",
                "base64_length": len(img_str),
                "estimated_size_kb": len(img_str) * 3 // 4 // 1024,
                "filename": filename
            },
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "height": request.height,
                "width": request.width,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            }
        }
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Add endpoint to test without returning full image
@app.post("/txt2img-test")
async def text_to_image_test(request: TextToImageRequest):
    try:
        print(f"TEST: Received request: {request.prompt[:50]}...")
        
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        # Generate image
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        ).images[0]
        
        # Save image locally for verification
        image.save("/tmp/test_output.png")
        
        # Return success without base64 (for testing)
        return {
            "status": "success",
            "message": "Image generated and saved to /tmp/test_output.png",
            "image_preview": f"data:image/png;base64,{base64.b64encode(image.tobytes())[:100].decode()}...",
            "parameters": request.dict()
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a simple test endpoint
@app.get("/")
async def root():
    return {"message": "Stable Diffusion API is running!", "endpoints": ["/health", "/txt2img"]}

# Add a quick test endpoint
@app.post("/test")
async def test_endpoint():
    return {"status": "API is working", "message": "Test successful"}

@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.1f} GB"
        }
    else:
        gpu_info = {"gpu_available": False}
    
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        **gpu_info
    }

if __name__ == "__main__":
    # RunPod typically exposes services on port 8000
    port = int(os.environ.get("PORT", 8000))
    # Bind to 0.0.0.0 to accept external connections
    uvicorn.run("app:app", host="0.0.0.0", port=port)
