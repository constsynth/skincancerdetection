import tensorflow as tf
from model_cnn import SKiN_CNN
import keras
import warnings

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
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), "acc"])
    model_name = 'SKIN_CNN_V3'
    model.fit(train_data, epochs=10, validation_data=test_data)
    model.save(f"./models/{model_name}.keras")

if __name__ == "__main__":
    # print(torch.cuda.is_available())

    # print(tf.config.list_physical_devices('GPU'))

    from tensorflow.python.client import device_lib


    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]

    
    print(get_available_devices())











