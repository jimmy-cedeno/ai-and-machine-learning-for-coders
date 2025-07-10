from tensorflow.keras.applications.inception_v3 import InceptionV3
import urllib.request
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

import zipfile
import os

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None)

pre_trained_model.load_weights(weights_file)


for layer in pre_trained_model.layers:
  layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
# print("Last layer output shape: ", last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(
  optimizer=RMSprop(learning_rate=0.0001),
  loss='binary_crossentropy',
  metrics=['accuracy']
)



train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# training_dir = 'horse-or-human/training/'
# validation_dir = 'horse-or-human/validation/'
training_url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
training_file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(training_url, training_file_name)
zip_ref = zipfile.ZipFile(training_file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

validation_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(validation_url, validation_file_name)

zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()





test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(training_dir,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=10,
                                                         class_mode='binary',
                                                         target_size=(150, 150))

print("Clases de entrenamiento:", train_generator.class_indices)
print("Clases de validación:", validation_generator.class_indices)
print("Cantidad de imágenes entrenamiento:", train_generator.samples)
print("Cantidad de imágenes validación:", validation_generator.samples)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    verbose=1)


# Test the model
test_path = "horse-or-human/test/"

for fn in os.listdir(test_path):
    path = os.path.join(test_path, fn)
    img = image.load_img(path, target_size=(150, 150))
    imgx = image.img_to_array(img)
    imgx = np.expand_dims(imgx, axis=0)
    imgx = imgx / 255.0

    image_tensor = np.vstack([imgx])
    classes = model.predict(image_tensor)

    print(classes)
    print(classes[0])

    if classes[0]>0.5:
        print(f'{fn} is a human')
    else:
        print(f'{fn} is a horse')