from train import train_model
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from image_processing import crop_center, super_resolution
model = tf.keras.models.load_model("./models/SKiN_CNN_2.keras")
print("Loaded model is SKiN_CNN_2.keras")

def inference_model(image: Image.Image) -> float:
    image = crop_center(image, frac=0.25)
    image = super_resolution(image)
    image = keras.utils.img_to_array(image)
    image = image/255.0
    prob = model.predict(np.expand_dims(image, 0))
    return round(prob[0][0], 2)

# if __name__ == "__main__":
#     # train_model() #if not pretrained yet
#     probability = inference_model(Image.open('datasets/train/malignant/20.jpg'))
#     print(probability)




