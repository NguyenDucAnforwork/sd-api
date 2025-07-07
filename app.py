import io
import torch
import base64
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

# Initialize the model
@app.on_event("startup")
async def startup_event():
    global pipe
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        # Use CPU with lower precision if GPU is not available
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
        # Set a seed if provided
        if request.seed is not None:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(request.seed)
        else:
            generator = None
            
        # Generate the image
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success",
            "image": img_str,
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
