from PaddleOCR.tools.infer import predict_system
import PaddleOCR.tools.infer.utility as utility
from PaddleOCR.ppocr.utils.logging import get_logger
from pathlib import Path
from typing import List
import shutil

logger = get_logger("validate")

IMAGES = Path("images")
INFERENCE = IMAGES.joinpath("inference")
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]


if __name__ == "__main__":
    args = utility.parse_args()
    args.use_onnx = True
    args.det_model_dir = "onnx/PP-OCRv5_mobile_giaa_det.onnx"
    args.rec_model_dir = "onnx/PP-OCRv5_mobile_giaa_rec.onnx"
    args.rec_char_dict_path = "configs/ppocrv5_dict.txt"

    if not IMAGES.exists():
        IMAGES.mkdir(exist_ok=True)

    if INFERENCE.exists():
        shutil.rmtree(INFERENCE)

    images: List[Path] = []

    for image_type in IMAGE_TYPES:
        images.extend(list(IMAGES.glob(f"*.{image_type}")))

    for image in images:
        logger.info(f"Processing {image}")
        filename = image.name
        args.draw_img_save_dir = str(INFERENCE.joinpath(f"{filename}"))
        args.image_dir = str(image)
        predict_system.main(args)
