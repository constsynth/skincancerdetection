import tensorflow as tf
class SKiN_CNN():
    def __init__(self):
        super().__init__()
        self.optimizers = ['adam']

        self.data_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=(256,256,3)),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomRotation(0.2)
    ])

        self.model = tf.keras.models.Sequential([
        self.data_augmentation,
        tf.keras.layers.Conv2D(16, (3,3), 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(16, (3,3), 1, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (5,5), 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(32, (5,5), 1, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (3,3), 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(16, (3,3), 1, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])





