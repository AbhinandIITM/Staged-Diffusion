import os
import re
import json
import random
import requests
import numpy as np
import cv2
import torch
import gradio as gr
from PIL import Image, ImageDraw

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionInpaintPipeline
)
from google import genai


# ==============================
# 🔧 Configuration
# ==============================
SAVE_DIR = "sd-1.5-10"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
W, H = 512, 512

client = genai.Client(api_key="YOUR_GEMINI_API_KEY")  # <-- Replace with env var for safety


# ==============================
# 🔹 Utility Functions
# ==============================
def safe_json_extract(text: str) -> dict | None:
    """Extract and parse JSON from raw text response."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print("⚠️ Could not parse extracted JSON.")
    return None


def concept_list_to_prompt(concept_list) -> str:
    """Convert concept list/dict to a text prompt string."""
    if isinstance(concept_list, str):
        return concept_list
    if isinstance(concept_list, list):
        parts = []
        for item in concept_list:
            if isinstance(item, dict):
                obj = item.get("object", "")
                desc = item.get("description", "")
                parts.append(f"{desc} {obj}".strip())
            else:
                parts.append(str(item))
        return ", ".join(parts)
    return str(concept_list)


# ==============================
# 🔹 Prompt Decomposition
# ==============================
def decompose_prompt_with_gemini(prompt: str) -> dict:
    """Use Gemini to split a scene into background, midground, foreground."""
    system_instruction = (
        "Given the scene description below, break it into three parts: "
        "background, midground, and foreground. Respond in JSON format."
    )
    full_prompt = f"{system_instruction}\n\nScene: \"{prompt}\""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=full_prompt
        )
        raw = response.text.strip()
        result = safe_json_extract(raw)
        if result:
            return result
        print("⚠️ Gemini output not valid JSON:\n", raw)
    except Exception as e:
        print("❌ Gemini decomposition failed:", e)

    # fallback heuristic
    phrases = [p.strip() for p in prompt.split(",") if p.strip()]
    return {
        "background": phrases[0] if len(phrases) > 0 else "",
        "midground": phrases[1] if len(phrases) > 1 else "",
        "foreground": phrases[2] if len(phrases) > 2 else ""
    }


# ==============================
# 🔹 Mask Utilities
# ==============================
def circular_mask(center: tuple[int, int], radius: int, size=(W, H)) -> Image.Image:
    """Return circular binary mask."""
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse(
        (center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), fill=255
    )
    return mask


def rectangular_mask(top_left: tuple[int, int], bottom_right: tuple[int, int], size=(W, H)) -> Image.Image:
    """Return rectangular binary mask."""
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle([top_left, bottom_right], fill=255)
    return mask


def draw_debug_layout(image: Image.Image, layout: dict) -> None:
    """Draw rectangles/circles for layout debugging."""
    draw = ImageDraw.Draw(image)
    for name, cfg in layout.items():
        if "center" in cfg and "radius" in cfg:
            x, y, r = *cfg["center"], cfg["radius"]
            draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=2)
            draw.text((x-r, y-r-10), name, fill="red")
        elif "top_left" in cfg and "bottom_right" in cfg:
            draw.rectangle([cfg["top_left"], cfg["bottom_right"]], outline="red", width=2)
            draw.text((cfg["top_left"][0], cfg["top_left"][1]-10), name, fill="red")
    image.save(f"{SAVE_DIR}/layout_debug.png")


# ==============================
# 🔹 Image Preprocessing
# ==============================
def canny_from_image(image: Image.Image, low=100, high=200) -> Image.Image:
    """Convert input image to Canny edge RGB image."""
    img_resized = image.resize((W, H))
    edges = cv2.Canny(np.array(img_resized.convert("RGB")), low, high)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


# ==============================
# 🔹 Stable Diffusion Pipelines
# ==============================
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch_dtype
)
text2img_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch_dtype
).to(device)
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch_dtype
).to(device)

# Enable optimizations
for pipe in [text2img_pipe, inpaint_pipe]:
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()


def generate_background(prompt: str) -> Image.Image:
    """Generate background layer with ControlNet."""
    blank = Image.new("RGB", (W, H), (255, 255, 255))
    canny = canny_from_image(blank)
    return text2img_pipe(prompt=prompt, image=canny,
                         num_inference_steps=20, guidance_scale=7.5).images[0]


def inpaint_layer(prompt: str, base_image: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply inpainting over masked region."""
    return inpaint_pipe(
        prompt=prompt,
        image=base_image.resize((W, H)),
        mask_image=mask.resize((W, H)).convert("L"),
        num_inference_steps=20,
        guidance_scale=8.0
    ).images[0]


# ==============================
# 🔹 Layered Generation
# ==============================
def layered_generation(prompt: str) -> None:
    """Generate background → midground → foreground layers and save them."""
    concepts = decompose_prompt_with_gemini(prompt)

    # Debug prints
    for key in ["background", "midground", "foreground"]:
        print(f"{key.capitalize()} prompt: {concepts.get(key)}")

    # Define simple rectangular masks
    layout = {
        "midground": {"top_left": (int(W*0.1), int(H*0.1)), "bottom_right": (int(W*0.9), int(H*0.9))},
        "foreground": {"top_left": (int(W*0.25), int(H*0.25)), "bottom_right": (int(W*0.75), int(H*0.75))}
    }
    draw_debug_layout(Image.new("RGB", (W, H), "white"), layout)

    # Generate layers
    background = generate_background(concepts["background"])
    background.save(f"{SAVE_DIR}/layer1_background.png")

    midground = inpaint_layer(concepts["midground"], background, rectangular_mask(**layout["midground"]))
    midground.save(f"{SAVE_DIR}/layer2_midground.png")

    final = inpaint_layer(concepts["foreground"], midground, rectangular_mask(**layout["foreground"]))
    final.save(f"{SAVE_DIR}/layer3_final.png")

    print(f"✅ All layers saved to {SAVE_DIR}")


# ==============================
# 🔹 Gradio Interface
# ==============================
def gradio_layered_generation(prompt: str):
    layered_generation(prompt)
    return (
        Image.open(f"{SAVE_DIR}/layer1_background.png"),
        Image.open(f"{SAVE_DIR}/layer2_midground.png"),
        Image.open(f"{SAVE_DIR}/layer3_final.png"),
    )


iface = gr.Interface(
    fn=gradio_layered_generation,
    inputs=gr.Textbox(lines=2, placeholder="Enter your scene prompt..."),
    outputs=[gr.Image(), gr.Image(), gr.Image()],
    title="Layered Image Generation",
    description="Decomposes a scene prompt into background, midground, and foreground, and generates each layer."
)

if __name__ == "__main__":
    iface.launch()
