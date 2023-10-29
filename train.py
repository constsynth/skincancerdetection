import tensorflow as tf
from models.model import SKiN_CNN
import keras
import warnings
import requests
import os
import zipfile
import io
def train_model():
    warnings.filterwarnings("ignore")
    tf.keras.backend.clear_session()
    print(tf.config.list_physical_devices('GPU'))

    path_to_datasets = "./datasets"
    # os.makedirs(path_to_datasets, exist_ok=True)
    # url = "https://drive.google.com/drive/folders/1U1eITyCjZeCgxJC-wEvXFHng05M8XBGO"
    # file = requests.get(url, allow_redirects=True)
    # z = zipfile.ZipFile(io.BytesIO(file.content))
    # z.extractall(path_to_datasets)
    # print('Dataset is downloaded')

    test_data = keras.utils.image_dataset_from_directory(f'{path_to_datasets}/test')
    train_data = keras.utils.image_dataset_from_directory(f'{path_to_datasets}/train')

    train_data = train_data.map(lambda x,y: (x/255, y))
    test_data = test_data.map(lambda x,y: (x/255, y))

    batch = train_data.as_numpy_iterator().next()
    print("Minimum value of the scaled data:", batch[0].min())
    print("Maximum value of the scaled data:", batch[0].max())


    init = SKiN_CNN()
    model = init.model # initizalize the model
    optimizer = init.optimizers[0] # using Adam optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])

    model.fit(train_data, epochs=10, validation_data=test_data)
    model.save("./models/SKiN_CNN_2.keras")

if __name__ == "__main__":
    train_model()













