from PIL import Image
import numpy as np
from tensorflow import keras
from cspackage.image_processing import crop_center, super_resolution

path_to_model = "./models/SKiN_CNN.keras"
model = keras.models.load_model(path_to_model)
print(f"Loaded model is {path_to_model}")

def inference_model(image: Image.Image) -> float:
    image = crop_center(image, frac=0.4)
    image = super_resolution(image)
    # image = image.resize((256, 256))
    print(image.size)
    image = keras.utils.img_to_array(image)
    image = image/255.0
    prob = model.predict(np.expand_dims(image, 0))
    return round(prob[0][0], 2)

# if __name__ == "__main__":
#     # train_model() #if not pretrained yet
#     probability = inference_model(Image.open('datasets/test/benign/216.jpg'))
#     print(probability)




