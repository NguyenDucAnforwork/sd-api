# API Usage Guide

## Base URL
Replace `your-pod-id` with your actual RunPod ID:
```
https://your-pod-id-8000.proxy.runpod.net
```

## 1. Test API Connection

### Using curl:
```bash
curl "https://your-pod-id-8000.proxy.runpod.net/"
```

### Using Postman:
- Method: `GET`
- URL: `https://your-pod-id-8000.proxy.runpod.net/`
- Send request

Expected response:
```json
{
  "message": "Stable Diffusion API is running!",
  "endpoints": ["/health", "/txt2img"]
}
```

## 2. Check Health Status

### Using curl:
```bash
curl "https://your-pod-id-8000.proxy.runpod.net/health"
```

### Using Postman:
- Method: `GET`
- URL: `https://your-pod-id-8000.proxy.runpod.net/health`

## 3. Generate Image

### Using curl:
```bash
curl -X POST "https://your-pod-id-8000.proxy.runpod.net/txt2img" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful mountain landscape, photorealistic, 4k",
    "negative_prompt": "blurry, low quality",
    "height": 512,
    "width": 512,
    "num_inference_steps": 20,
    "guidance_scale": 7.5
  }'
```

### Using Postman:

1. **Method**: `POST`
2. **URL**: `https://your-pod-id-8000.proxy.runpod.net/txt2img`
3. **Headers**: 
   - Key: `Content-Type`
   - Value: `application/json`
4. **Body**: Select `raw` and `JSON`, then paste:
   ```json
   {
     "prompt": "A beautiful mountain landscape, photorealistic, 4k",
     "negative_prompt": "blurry, low quality",
     "height": 512,
     "width": 512,
     "num_inference_steps": 20,
     "guidance_scale": 7.5
   }
   ```
5. **Send** the request

## 4. Decode Base64 Image

The API returns the image as base64 string. To view it:

### Method 1: Save to file (Linux/Mac)
```bash
# Save the base64 string to a file and decode
echo "YOUR_BASE64_STRING" | base64 -d > output.png
```

### Method 2: Online decoder
1. Copy the base64 string from the response
2. Go to https://base64.guru/converter/decode/image
3. Paste and decode

### Method 3: Python script
```python
import base64
from PIL import Image
import io

# Your base64 string from API response
base64_string = "YOUR_BASE64_STRING"

# Decode and save
image_data = base64.b64decode(base64_string)
image = Image.open(io.BytesIO(image_data))
image.save("output.png")
```

## 5. Troubleshooting

### If you get no response:
1. Check if the pod is running: `https://your-pod-id-8000.proxy.runpod.net/health`
2. Try the test endpoint first: `https://your-pod-id-8000.proxy.runpod.net/test`
3. Check RunPod logs for errors

### If you get timeout:
1. Reduce `num_inference_steps` to 10-20
2. Use smaller image dimensions (256x256)
3. Check GPU memory usage

### Common curl issues:
- Make sure to use double quotes for JSON
- Escape special characters in the prompt
- Add `--max-time 300` for longer timeout:
  ```bash
  curl --max-time 300 -X POST "https://your-pod-id-8000.proxy.runpod.net/txt2img" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "A cat", "num_inference_steps": 10}'
  ```

## 6. Example Complete Workflow

1. **Test connection**:
   ```bash
   curl "https://abc123-8000.proxy.runpod.net/"
   ```

2. **Check health**:
   ```bash
   curl "https://abc123-8000.proxy.runpod.net/health"
   ```

3. **Generate simple image**:
   ```bash
   curl -X POST "https://abc123-8000.proxy.runpod.net/txt2img" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A cute cat", "num_inference_steps": 10}'
   ```

4. **Save response to file**:
   ```bash
   curl -X POST "https://abc123-8000.proxy.runpod.net/txt2img" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A cute cat", "num_inference_steps": 10}' \
     > response.json
   ```
