import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import spaces
import gradio as gr
import typer
import torch
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI
from loguru import logger
from typer import Option
from typer_config import use_json_config

from iopaint.const import *
from iopaint.runtime import setup_model_dir, dump_environment_info, check_device
from iopaint.schema import InteractiveSegModel, Device, RealESRGANModel, RemoveBGModel, InpaintRequest
from iopaint.model_manager import ModelManager
from iopaint.model.utils import torch_gc
from iopaint.helper import decode_base64_to_image, concat_alpha_channel

typer_app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@typer_app.command(help="Install all plugins dependencies")
def install_plugins_packages():
    from iopaint.installer import install_plugins_package

    install_plugins_package()


@typer_app.command(help="Download SD/SDXL normal/inpainting model from HuggingFace")
def download(
    model: str = Option(
        ..., help="Model id on HuggingFace e.g: runwayml/stable-diffusion-inpainting"
    ),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import cli_download_model

    cli_download_model(model)


@typer_app.command(name="list", help="List downloaded models")
def list_model(
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import scan_models

    scanned_models = scan_models()
    for it in scanned_models:
        print(it.name)


@typer_app.command(help="Batch processing images")
def run(
    model: str = Option("lama"),
    device: Device = Option(Device.cpu),
    image: Path = Option(..., help="Image folders or file path"),
    mask: Path = Option(
        ...,
        help="Mask folders or file path. "
        "If it is a directory, the mask images in the directory should have the same name as the original image."
        "If it is a file, all images will use this mask."
        "Mask will automatically resize to the same size as the original image.",
    ),
    output: Path = Option(..., help="Output directory or file path"),
    config: Path = Option(
        None, help="Config file path. You can use dump command to create a base config."
    ),
    concat: bool = Option(
        False, help="Concat original image, mask and output images into one image"
    ),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import cli_download_model, scan_models

    scanned_models = scan_models()
    if model not in [it.name for it in scanned_models]:
        logger.info(f"{model} not found in {model_dir}, try to downloading")
        cli_download_model(model)

    from iopaint.batch_processing import batch_inpaint

    batch_inpaint(model, device, image, mask, output, config, concat)


@typer_app.command(help="Start IOPaint server")
@use_json_config()
def start(
    host: str = Option("127.0.0.1"),
    port: int = Option(8080),
    inbrowser: bool = Option(False, help=INBROWSER_HELP),
    model: str = Option(
        DEFAULT_MODEL,
        help=f"Erase models: [{', '.join(AVAILABLE_MODELS)}].\n"
        f"Diffusion models: [{', '.join(DIFFUSION_MODELS)}] or any SD/SDXL normal/inpainting models on HuggingFace.",
    ),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        dir_okay=True,
        file_okay=False,
        callback=setup_model_dir,
    ),
    low_mem: bool = Option(False, help=LOW_MEM_HELP),
    no_half: bool = Option(False, help=NO_HALF_HELP),
    cpu_offload: bool = Option(False, help=CPU_OFFLOAD_HELP),
    disable_nsfw_checker: bool = Option(False, help=DISABLE_NSFW_HELP),
    cpu_textencoder: bool = Option(False, help=CPU_TEXTENCODER_HELP),
    local_files_only: bool = Option(False, help=LOCAL_FILES_ONLY_HELP),
    device: Device = Option(Device.cpu),
    input: Optional[Path] = Option(None, help=INPUT_HELP),
    mask_dir: Optional[Path] = Option(
        None, help=MODEL_DIR_HELP, dir_okay=True, file_okay=False
    ),
    output_dir: Optional[Path] = Option(
        None, help=OUTPUT_DIR_HELP, dir_okay=True, file_okay=False
    ),
    quality: int = Option(100, help=QUALITY_HELP),
    enable_interactive_seg: bool = Option(False, help=INTERACTIVE_SEG_HELP),
    interactive_seg_model: InteractiveSegModel = Option(
        InteractiveSegModel.sam2_1_tiny, help=INTERACTIVE_SEG_MODEL_HELP
    ),
    interactive_seg_device: Device = Option(Device.cpu),
    enable_remove_bg: bool = Option(False, help=REMOVE_BG_HELP),
    remove_bg_device: Device = Option(Device.cpu, help=REMOVE_BG_DEVICE_HELP),
    remove_bg_model: RemoveBGModel = Option(RemoveBGModel.briaai_rmbg_1_4),
    enable_anime_seg: bool = Option(False, help=ANIMESEG_HELP),
    enable_realesrgan: bool = Option(False),
    realesrgan_device: Device = Option(Device.cpu),
    realesrgan_model: RealESRGANModel = Option(RealESRGANModel.realesr_general_x4v3),
    enable_gfpgan: bool = Option(False),
    gfpgan_device: Device = Option(Device.cpu),
    enable_restoreformer: bool = Option(False),
    restoreformer_device: Device = Option(Device.cpu),
):
    dump_environment_info()
    device = check_device(device)
    remove_bg_device = check_device(remove_bg_device)
    realesrgan_device = check_device(realesrgan_device)
    gfpgan_device = check_device(gfpgan_device)

    if input and not input.exists():
        logger.error(f"invalid --input: {input} not exists")
        exit(-1)
    if mask_dir and not mask_dir.exists():
        logger.error(f"invalid --mask-dir: {mask_dir} not exists")
        exit(-1)
    if input and input.is_dir() and not output_dir:
        logger.error(
            "invalid --output-dir: --output-dir must be set when --input is a directory"
        )
        exit(-1)
    if output_dir:
        output_dir = output_dir.expanduser().absolute()
        logger.info(f"Image will be saved to {output_dir}")
        if not output_dir.exists():
            logger.info(f"Create output directory {output_dir}")
            output_dir.mkdir(parents=True)
    if mask_dir:
        mask_dir = mask_dir.expanduser().absolute()

    model_dir = model_dir.expanduser().absolute()

    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from iopaint.download import cli_download_model, scan_models

    scanned_models = scan_models()
    if model not in [it.name for it in scanned_models]:
        logger.info(f"{model} not found in {model_dir}, try to downloading")
        cli_download_model(model)

    from iopaint.api import Api
    from iopaint.schema import ApiConfig

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if inbrowser:
            webbrowser.open(f"http://localhost:{port}", new=0, autoraise=True)
        yield

    app = FastAPI(lifespan=lifespan)

    api_config = ApiConfig(
        host=host,
        port=port,
        inbrowser=inbrowser,
        model=model,
        no_half=no_half,
        low_mem=low_mem,
        cpu_offload=cpu_offload,
        disable_nsfw_checker=disable_nsfw_checker,
        local_files_only=local_files_only,
        cpu_textencoder=cpu_textencoder if device == Device.cuda else False,
        device=device,
        input=input,
        mask_dir=mask_dir,
        output_dir=output_dir,
        quality=quality,
        enable_interactive_seg=enable_interactive_seg,
        interactive_seg_model=interactive_seg_model,
        interactive_seg_device=interactive_seg_device,
        enable_remove_bg=enable_remove_bg,
        remove_bg_device=remove_bg_device,
        remove_bg_model=remove_bg_model,
        enable_anime_seg=enable_anime_seg,
        enable_realesrgan=enable_realesrgan,
        realesrgan_device=realesrgan_device,
        realesrgan_model=realesrgan_model,
        enable_gfpgan=enable_gfpgan,
        gfpgan_device=gfpgan_device,
        enable_restoreformer=enable_restoreformer,
        restoreformer_device=restoreformer_device,
    )
    print(api_config.model_dump_json(indent=4))
    api = Api(app, api_config)
    api.launch()


@typer_app.command(name="gradio")
def start_gradio(
    device: Device = Option(Device.cpu, help="Device to run the model on"),
    quality: int = Option(100, help="Output image quality"),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        file_okay=False,
        callback=setup_model_dir,
    ),
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

@typer_app.command(help="Start IOPaint web config page")
def start_web_config(
    config_file: Path = Option("config.json"),
):
    dump_environment_info()
    from iopaint.web_config import main

    main(config_file)
