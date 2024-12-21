import modal

# Create a Modal stub (a container for functions and resources)
stub = modal.Stub("pentagram-backend")

# Define a Modal image with the required dependencies for Stable Diffusion
image = modal.Image.debian_slim().pip_install(["torch", "diffusers", "transformers"])

# Define the function to load the Stable Diffusion model
@stub.function(image=image, gpu="any", timeout=300)
def generate_image(prompt: str) -> bytes:
    """
    Generates an image based on a text prompt using a Stable Diffusion model.
    Args:
        prompt (str): The text description for the image to generate.
    Returns:
        bytes: The generated image in byte format.
    """
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
