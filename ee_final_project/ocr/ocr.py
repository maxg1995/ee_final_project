import asyncio
import os

import PIL
import torch
from PIL import Image, ImageFile
from torchvision import transforms

from ee_final_project.env import MODEL_PATH, SAMPLES_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

def OCR():
    print('strting OCR')
    model = torch.load(MODEL_PATH)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5), std=(0.5))])
    
    images = []
    for image_path in os.listdir(SAMPLES_DIR):
        if image_path != '.DS_Store':
            try:
                image = Image.open(os.path.join(SAMPLES_DIR, image_path))
                image = transform(image)
                images.append(image)
            except PIL.UnidentifiedImageError:
                print("cannotidentify")

    results = []
    for i, image in enumerate(images):
        num_of_digits = int(image.shape[2] / 28)

        # Split the n-digit image into n same equal parts
        res = ''
        for i in range(num_of_digits):
            img = image[0, :, :].view(28, 28 * num_of_digits)
            single_digit = img[:, (i)*28:(i+1)*28]
            single_digit_reshaped = single_digit.reshape(1, 28*28)

            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(single_digit_reshaped)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            res += str(probab.index(max(probab)))
        
        results.append(res)

        if i % 100 == 0:
            print(f"finished {i} images")
            
    return results

async def convert_images_to_text():
    file_names = os.listdir(SAMPLES_DIR)
    result = await asyncio.gather(
        *map(
            ocr,
            file_names,
        )
    )
    return result
