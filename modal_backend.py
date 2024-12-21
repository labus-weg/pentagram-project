import modal

# Create a Modal application
app = modal.App(name="pentagram-backend")

# Define the function to generate an image
@app.function()
def generate_image(prompt: str) -> bytes:
    from diffusers import StableDiffusionPipeline
    import torch
    from io import BytesIO

    # Load the Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline.to("cuda")  # Move model to GPU for faster inference

    # Generate image
    image = pipeline(prompt).images[0]

    # Convert the image to bytes for transfer
    byte_io = BytesIO()
    image.save(byte_io, format="PNG")
    return byte_io.getvalue()
