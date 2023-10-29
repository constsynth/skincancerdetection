from train import train_model
from PIL import Image
from model.wdsr import wdsr_b
import numpy as np
from tensorflow import convert_to_tensor
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from model import resolve_single
from model.edsr import edsr
from utils import load_image, plot_sample
from model.srgan import generator

# path_to_datasets = "./datasets"
# test_data = keras.utils.image_dataset_from_directory(f'{path_to_datasets}/test')
# test_data = test_data.map(lambda x,y: (x/255, y))
model = tf.keras.models.load_model("./models/SKiN_CNN_2.keras")
print("Loaded model is SKiN_CNN_2.keras")

model_sr_gan = generator(num_filters=64, num_res_blocks=16)
model_sr_gan.load_weights('./models/gan_generator.h5')

model_sr_32_x4 = wdsr_b(scale=4, num_res_blocks=32)
model_sr_32_x4.load_weights('./models/weights.h5')

model_sr_16_x4 = edsr(scale=4, num_res_blocks=16)
model_sr_16_x4.load_weights('./models/weights_16.h5')

def inference_model(image: Image.Image) -> float:
    frac = 0.35
    left = image.size[0] * ((1 - frac) / 2)
    upper = image.size[1] * ((1 - frac) / 2)
    right = image.size[0] - ((1 - frac) / 2) * image.size[0]
    bottom = image.size[1] - ((1 - frac) / 2) * image.size[1]
    cropped_img = image.crop((left, upper, right, bottom))
    cropped_img = cropped_img.resize((64, 64))

    sr_16_x4 = resolve_single(model_sr_16_x4, cropped_img)
    sr_32_x4 = resolve_single(model_sr_32_x4, cropped_img)
    sr_gan = resolve_single(model_sr_gan, cropped_img)

    print(cropped_img.size)
    fig = plt.figure(figsize=(10, 10))
    rows = 1
    columns = 4
    fig.add_subplot(rows, columns, 1)
    plt.imshow(cropped_img)
    plt.axis('off')
    plt.title("non sr")

    fig.add_subplot(rows, columns, 2)
    # showing image

    plt.imshow(sr_16_x4)
    plt.axis('off')
    plt.title("sr_16_x4")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(sr_32_x4)
    plt.axis('off')
    plt.title("sr_32_x4")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(sr_gan)
    plt.axis('off')
    plt.title("sr_gan")
    plt.show()

    # plt.imshow(cropped_img)
    # plt.show()
    # plt.imshow(sr)
    # plt.show()
    # image = image.resize((256, 256))
    # image = keras.utils.img_to_array(image)
    # image = image/255.0
    # prob = model.predict(np.expand_dims(image, 0))
    # return round(prob[0][0], 2)

if __name__ == "__main__":
    # train_model() #if not pretrained yet
    probability = inference_model(Image.open('benign-growthsfig5_large.jpg'))
#     print(probability)




