import io
import cv2
import json
import time
import typer
import runpod
import uvicorn
import webbrowser
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Optional
from typer import Option
from loguru import logger
from pydantic import Field
from fastapi.responses import Response
from typer_config import use_json_config
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Header, File, Form, HTTPException

from iopaint.const import *
from iopaint.gcs_service import gcs_handler
from iopaint.runtime import setup_model_dir, dump_environment_info, check_device
from iopaint.model_manager import ModelManager
from iopaint.model.utils import torch_gc
from iopaint.download import cli_download_model, scan_models
from iopaint.helper import decode_base64_to_image, get_image_ext, concat_alpha_channel, pil_to_bytes
from iopaint.schema import Choices, InteractiveSegModel, Device, RealESRGANModel, RemoveBGModel, InpaintRequest

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


class RunpodModelName(Choices):
    lama = "lama"
    anime_lama = "anime-lama"
    
class RunpodInpaintRequest(InpaintRequest):
    model_name: RunpodModelName = Field(RunpodModelName.lama, description="Model name to use for inpainting")

def process_uploaded_image(image_bytes, gray=False):
    """Convert uploaded image to numpy array"""
    ext = get_image_ext(image_bytes)
    image = Image.open(io.BytesIO(image_bytes))
    
    alpha_channel = None
    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass
    infos = image.info
    
    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)
    
    return np_img, alpha_channel, infos, ext

def inpaint_process(
    model_manager: ModelManager,
    image_bytes: bytes,
    mask_bytes: bytes,
    req: InpaintRequest
):
    t1 = time.time()
    # Convert uploaded files thành numpy arrays
    np_img, alpha_channel, infos, ext = process_uploaded_image(image_bytes)
    np_mask, _, _, _ = process_uploaded_image(mask_bytes, gray=True)
    
    # Xử lý mask
    np_mask = cv2.threshold(np_mask, 127, 255, cv2.THRESH_BINARY)[1]
    if np_img.shape[:2] != np_mask.shape[:2]:
        raise HTTPException(
            400,
            detail=f"Image size({np_img.shape[:2]}) and mask size({np_mask.shape[:2]}) not match.",
        )
    
    # Xử lý inpainting
    start = time.time()
    rgb_np_img = model_manager(np_img, np_mask, req)
    logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
    torch_gc()
    
    # Convert kết quả và trả về
    rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)
    time_taken = round(time.time() - t1, 3)
    return rgb_res, infos, ext, time_taken

@typer_app.command(name="runpod")
def init_runpod_worker(device: Device = Option(Device.cpu)):
    dump_environment_info()
    device = check_device(device)
    logger.info(f"Using device: {device}")
    
    scanned_models = scan_models()
    for model in ["lama", "anime-lama"]:
        if model not in [it.name for it in scanned_models]:
            logger.info(f"{model} not found, try to downloading")
            cli_download_model(model)
    
    lama_model_manager = ModelManager(
        name="lama",
        device=device,
        no_half=False,
        low_mem=False
    )
    anime_lama_model_manager = ModelManager(
        name="anime-lama",
        device=device,
        no_half=False,
        low_mem=False
    )
    logger.info("Model initialized")
    
    def handler(event):
        """
        This function processes incoming requests to your Serverless endpoint.
        
        Args:
            event (dict): Contains the input data and request metadata
            
        Returns:
            Any: The result to be returned to the client
        """
        input = event["input"]
        
        try:
            req = RunpodInpaintRequest(**input)
        except Exception as e:
            return {
                "error": f"Invalid request data: {str(e)}"
            }
        image_name = req.image
        mask_name = req.mask
        image_bytes, image_time = gcs_handler.download_image(image_name)
        mask_bytes, mask_time = gcs_handler.download_image(mask_name)
        
        model_manager = lama_model_manager if req.model_name == "lama" else anime_lama_model_manager
        rgb_res, infos, ext, inpain_time = inpaint_process(model_manager, image_bytes, mask_bytes, req)
        
        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=100,
            infos=infos,
        )
        dest_name, upload_time = gcs_handler.upload_image(res_img_bytes, "result." + ext)
        return {
            "dest_name": dest_name,
            "ext": ext,
            "image_time": image_time,
            "mask_time": mask_time,
            "inpain_time": inpain_time,
            "upload_time": upload_time,
        }

    # Start the Serverless function when the script is run
    runpod.serverless.start({
        "handler": handler
    })

@typer_app.command(name="inpaint-api")
def start_inpaint_api(device: Device = Option(Device.cpu)):
    API_KEY = "transwise-inpaint"
    dump_environment_info()
    device = check_device(device)
    logger.info(f"Using device: {device}")
    
    scanned_models = scan_models()
    for model in ["lama", "anime-lama"]:
        if model not in [it.name for it in scanned_models]:
            logger.info(f"{model} not found, try to downloading")
            cli_download_model(model)
    
    lama_model_manager = ModelManager(
        name="lama",
        device=device,
        no_half=False,
        low_mem=False
    )
    anime_lama_model_manager = ModelManager(
        name="anime-lama",
        device=device,
        no_half=False,
        low_mem=False
    )
    logger.info("Model initialized")
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(
        lifespan=lifespan,
        docs_url="/tw-docs",
        redoc_url="/tw-redoc",
    )
    
    @app.post("/inpaint")
    async def inpaint_dispatch(
        api_key: Optional[str] = Header(None),
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        request_data: str = Form(...),
    ):
        if not API_KEY or api_key != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        # Parse request body
        try:
            req = RunpodInpaintRequest(**json.loads(request_data))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request data: {str(e)}"
            )
        
        # Đọc file uploads
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        
        # Chọn model manager dựa trên model_name
        model_manager = lama_model_manager if req.model_name == "lama" else anime_lama_model_manager
        rgb_res, infos, ext = inpaint_process(model_manager, image_bytes, mask_bytes, req)
        
        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=100,
            infos=infos,
        )
        
        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
            headers={"X-Seed": str(req.sd_seed)},
        )
    
    uvicorn.run(app, host="0.0.0.0", port=8675)
    

@typer_app.command(help="Start IOPaint web config page")
def start_web_config(
    config_file: Path = Option("config.json"),
):
    dump_environment_info()
    from iopaint.web_config import main

    main(config_file)
    