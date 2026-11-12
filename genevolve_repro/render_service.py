import argparse
import base64
import binascii
from io import BytesIO
from threading import Lock

from PIL import Image, UnidentifiedImageError

from .common import source_lock


def decode_reference(data_url):
    header, separator, body = data_url.partition(",")
    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise ValueError("Expected an image data URL.")
    if len(body) > 24_000_000:
        raise ValueError("The reference image is too large.")
    try:
        with Image.open(BytesIO(base64.b64decode(body, validate=True))) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, binascii.Error, OSError) as error:
        raise ValueError("Invalid reference image.") from error


def app_for(pipeline):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import torch

    app = FastAPI()
    lock = Lock()

    class Request(BaseModel):
        prompt: str = Field(min_length=1)
        image_urls: list[str] = Field(min_length=1, max_length=2)
        width: int = Field(gt=0, le=2048)
        height: int = Field(gt=0, le=2048)
        seed: int = 0
        true_cfg_scale: float = 4.0
        negative_prompt: str = " "
        num_inference_steps: int = Field(default=40, ge=1, le=100)
        guidance_scale: float = 1.0
        num_images_per_prompt: int = Field(default=1, ge=1, le=1)

    @app.get("/health")
    def health():
        return {"ready": True, "model": "Qwen-Image-Edit-2511"}

    @app.post("/generate")
    def generate(request: Request):
        try:
            references = [decode_reference(url) for url in request.image_urls]
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            arguments = request.model_dump(exclude={"image_urls", "seed"})
            generator = torch.Generator(device="cpu").manual_seed(request.seed)
            with lock, torch.inference_mode():
                result = pipeline(image=references, generator=generator, **arguments)
            with BytesIO() as buffer:
                result.images[0].save(buffer, format="PNG")
                return {"success": True, "image": base64.b64encode(buffer.getvalue()).decode("ascii")}
        finally:
            for reference in references:
                reference.close()

    return app


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 renderer for GenEvolve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()
    from diffusers import QwenImageEditPlusPipeline
    import torch
    import uvicorn

    model = source_lock()["renderer"]
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model["repo"], revision=model["revision"], torch_dtype=torch.bfloat16,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    uvicorn.run(app_for(pipe), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
