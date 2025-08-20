# Staged Diffusion: Layered Image Generation

This project uses Stable Diffusion and ControlNet to generate images layer by layer, based on a user-provided prompt.  The prompt is decomposed into background, midground, and foreground components, and each layer is generated separately.

## Features

* **Layered Generation:** Generates images in stages (background, midground, foreground) for more control and realism.
* **Prompt Decomposition:** Uses the Gemini API to intelligently break down complex prompts into manageable parts.
* **ControlNet Integration:** Leverages ControlNet for precise control over the image generation process.
* **Stable Diffusion:** Utilizes the power of Stable Diffusion for high-quality image synthesis.

## Dependencies

* `numpy`
* `opencv-python`
* `torch`
* `diffusers`
* `gradio`
* `Pillow`
* `requests`
* `google-generativeai`

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Obtain Gemini API Key:**  Replace `"YOUR_GEMINI_API_KEY"` in `inpaint.py` with your actual Gemini API key.  Consider storing this securely as an environment variable.
3. **Run the Application:**
   ```bash
   python inpaint.py
   ```
   This will launch a Gradio interface where you can enter your prompt.

## Usage

1. Run the script.
2. Enter a scene description in the text box.  The more detailed your prompt, the better the results.  Consider using commas to separate different aspects of the scene.
3. Click "Submit".  The script will generate the background, midground, and foreground layers, saving them to the `sd-1.5-10` directory.  A debug image showing the mask layout will also be saved.

## Example Prompts

* `"a cozy cabin in a snowy forest, a person sitting by the fireplace"`
* `"a bustling city street at night, a lone figure walking down the street, a bright neon sign"`
* `"a futuristic spaceship flying through space, a planet in the distance, an astronaut looking out the window"`

## Directory Structure

The generated images are saved in the `sd-1.5-10` directory (this number may change).  The directory contains:

* `layer1_background.png`: The background layer.
* `layer2_midground.png`: The midground layer.
* `layer3_final.png`: The final foreground layer.
* `layout_debug.png`: A debug image showing the mask layout.


## Contributing

Contributions are welcome!  Please open an issue or submit a pull request.

## License

[MIT License](LICENSE)
