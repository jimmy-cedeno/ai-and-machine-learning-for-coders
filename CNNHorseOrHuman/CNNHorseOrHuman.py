import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.src.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
training_dir = 'horse-or-human/training/'
validation_dir = 'horse-or-human/validation/'

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              # there are only two classes, and this is a loss function that is designed for that scenario
              optimizer=RMSprop(learning_rate=0.001),
              # (Root Mean Square Propagation) that takes a learning rate parameter that allows us to tweak the learning
              metrics=['accuracy']
              )

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
)

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)


# Test the model
test_path = "horse-or-human/test/"

for fn in os.listdir(test_path):
    path = os.path.join(test_path, fn)
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)

    print(classes)
    print(classes[0])

    if classes[0] > 0.5:
        print(f'{fn} is a human')
    else:
        print(f'{fn} is a horse')