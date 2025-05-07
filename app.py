import time
from pathlib import Path

import spaces
import torch
import numpy as np
import gradio as gr
from PIL import Image
import cv2
from loguru import logger

from iopaint.const import *
from iopaint.runtime import setup_model_dir, dump_environment_info, check_device
from iopaint.schema import Device, InpaintRequest
from iopaint.model_manager import ModelManager
from iopaint.model.utils import torch_gc
from iopaint.helper import decode_base64_to_image, concat_alpha_channel

def start_gradio(
    device: Device = Device.cuda,
    quality: int = 100,
):
    """Start a Gradio app for anime-lama inpainting using base64 encoded images and masks."""
    dump_environment_info()
    device = check_device(device)
    logger.info(f"Using device: {device}")
    
    # Initialize the model on CPU first
    model_manager = ModelManager(
        name="anime-lama",
        device=torch.device("cpu"),
        no_half=False,
        low_mem=False
    )
    logger.info("Model initialized on CPU")
    
    # Create a request object template
    request_template = InpaintRequest()
    
    @spaces.GPU(duration=5)
    def process_inpaint(image_base64: str, mask_base64: str, hd_strategy: str):
        try:
            # Move model to GPU when processing
            if str(device) == "cuda":
                logger.info("Moving model to GPU")
                model_manager.device = torch.device("cuda")
            
            # Create request object
            req = request_template.model_copy()
            req.image = image_base64
            req.mask = mask_base64
            req.hd_strategy = hd_strategy
            
            # Process inpainting similar to api_inpaint in iopaint/api.py
            image, alpha_channel, infos, ext = decode_base64_to_image(req.image)
            mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
            logger.info(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
            
            # Apply threshold to mask
            mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            
            # Validate image and mask dimensions match
            if image.shape[:2] != mask.shape[:2]:
                return None, f"Error: Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) do not match."
            
            # Process with model
            logger.info("Running inpainting...")
            start = time.time()
            bgr_np_img = model_manager(image, mask, req)
            logger.info(f"Inpainting completed in {(time.time() - start) * 1000:.2f}ms")
            rgb_np_img = cv2.cvtColor(bgr_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            
            # Convert BGR to RGB and handle alpha channel
            result = concat_alpha_channel(rgb_np_img, alpha_channel)
            
            # Move model back to CPU
            if str(device) == "cuda":
                logger.info("Moving model back to CPU")
                model_manager.device = torch.device("cpu")
                torch_gc()

            
            return Image.fromarray(result), None
            
        except Exception as e:
            logger.error(f"Error during inpainting: {str(e)}")
            # Move model back to CPU in case of error
            if str(device) == "cuda":
                model_manager.device = torch.device("cpu")
                torch_gc()
            return None, f"Error: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks() as app:
        gr.Markdown("# Anime-LaMa Inpainting")
        gr.Markdown("### Nhập ảnh và mask dưới dạng base64")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.TextArea(label="Base64 Image")
                mask_input = gr.TextArea(label="Base64 Mask")
                hd_strategy = gr.Radio(
                    label="HD Strategy",
                    choices=["Original", "Resize", "Crop"],
                    value="Crop",
                    type="value",
                )
                with gr.Row():
                    submit_btn = gr.Button("Inpaint", variant="primary")
            
            with gr.Column():
                result_image = gr.Image(label="Result", type="pil")
                error_output = gr.Textbox(label="Error (if any)", visible=True)
        
        submit_btn.click(
            fn=process_inpaint,
            inputs=[image_input, mask_input, hd_strategy],
            outputs=[result_image, error_output]
        )
    
    # Launch the app
    app.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    start_gradio()
