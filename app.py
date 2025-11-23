import sys
import io
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from rembg import remove
from PIL import Image

# --- Setup Paths ---
sys.path.append("notebook")
from inference import Inference

# Initialize API
app = FastAPI()

# --- Load Model (Global) ---
TAG = "hf"
CONFIG_PATH = f"checkpoints/{TAG}/pipeline.yaml"

print("Loading 3D Inference Model...")
# compile=False is safer for Docker to avoid torch compile overhead on startup
inference_model = Inference(CONFIG_PATH, compile=False)
print("Model Loaded.")

@app.post("/generate-mesh")
async def generate_mesh(
    image_file: UploadFile = File(...),
    seed: int = Form(42)
):
    # 1. Read the uploaded image
    image_data = await image_file.read()
    input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # 2. Generate Mask (Background Removal)
    print("Generating mask...")
    # rembg removes background and returns RGBA
    no_bg_image = remove(input_image) 
    
    # Extract the Alpha channel to use as the mask
    # This creates a black/white image where the object is white
    mask = no_bg_image.split()[-1] 

    # 3. Handle File I/O for the Inference Class
    # Many research codebases (like sam3d) expect file paths, not PIL objects.
    # We create a temporary directory to store the inputs for the model to read.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_img_path = os.path.join(temp_dir, "input.png")
        temp_mask_path = os.path.join(temp_dir, "mask.png")
        output_ply_path = os.path.join(temp_dir, "output.ply")

        # Save inputs to disk so inference() can read them
        input_image.save(temp_img_path)
        mask.save(temp_mask_path)

        # 4. Run Inference
        # We pass the PATHS or PIL objects depending on what your inference.py expects.
        # Assuming based on your previous snippet it takes the loaded objects:
        # If your load_image functions are inside inference.py, we might use them here:
        
        # Scenario A: Passing PIL objects directly (If your inference supports it)
        # output = inference_model(input_image, mask, seed=seed)
        
        # Scenario B: Passing File Paths (Safer if logic relies on 'load_image')
        # We assume you adapted inference.py to take the objects, or we use the helper:
        from inference import load_image, load_single_mask 
        
        # Re-load from the temp paths to ensure format consistency with original code
        loaded_image = load_image(temp_img_path)
        
        # We can't use load_single_mask(index=...) anymore. 
        # You likely need to pass the mask object directly.
        # Assuming you modified inference() to accept a PIL mask or path:
        output = inference_model(loaded_image, mask, seed=seed)

        # 5. Export Mesh
        # Using the logic you provided
        mesh = output["glb"] 
        mesh.export(output_ply_path)

        # 6. Return the file
        # We return FileResponse which handles the file streaming automatically
        return FileResponse(
            output_ply_path, 
            media_type="application/octet-stream", 
            filename="result.ply"
        )