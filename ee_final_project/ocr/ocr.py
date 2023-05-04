import asyncio
import os

import numpy as np
import pytesseract
from env import SAMPLES_DIR
from PIL import Image


async def ocr(file_name: str):
    img = np.array(Image.open(f"{SAMPLES_DIR}/{file_name}"))
    text = pytesseract.image_to_string(img)

    return text


async def convert_images_to_text():
    file_names = os.listdir(SAMPLES_DIR)
    result = await asyncio.gather(
        *map(
            ocr,
            file_names,
        )
    )
    return result
