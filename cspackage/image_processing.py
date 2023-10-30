from PIL import Image
from model.srgan import generator
from model import resolve_single
import numpy as np
def crop_center(image: Image.Image,
                frac: float = None) -> Image.Image:
    frac = frac
    left = image.size[0] * ((1 - frac) / 2)
    upper = image.size[1] * ((1 - frac) / 2)
    right = image.size[0] - ((1 - frac) / 2) * image.size[0]
    bottom = image.size[1] - ((1 - frac) / 2) * image.size[1]
    cropped_img = image.crop((left, upper, right, bottom))
    cropped_img = cropped_img.resize((256, 256))
    cropped_img.save('./sr_not_applied.jpg')
    return cropped_img

def super_resolution(image: Image.Image) -> Image.Image:
    model_sr_gan = generator(num_filters=64, num_res_blocks=16)
    model_sr_gan.load_weights('./models/gan_generator.h5')
    image = resolve_single(model_sr_gan, image)
    image = np.array(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image = image.resize((256, 256))
    image.save('./sr_is_applied.jpg')
    return image
