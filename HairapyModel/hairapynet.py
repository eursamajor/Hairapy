#IMPORT DATA

import os
import shutil
import random
import zipfile
import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image, ImageEnhance, ImageOps

url = 'https://github.com/eursamajor/Hairapy/raw/main/Dataset.zip'
local_path = './Dataset.zip'
urllib.request.urlretrieve(url, local_path)

with zipfile.ZipFile(local_path, 'r') as zip_ref:
    zip_ref.extractall('data')

dataset_dir = './data/Dataset'
output_dir = './data/BalancedDataset'
os.makedirs(output_dir, exist_ok=True)

class_dirs = ['Dandruff', 'Hair Greasy', 'Hair Loss', 'Psoriasis']
class_counts = {class_name: len(os.listdir(os.path.join(dataset_dir, class_name))) for class_name in class_dirs}
max_count = max(class_counts.values())

#DATA PREPROCESSING
def random_rotation(image):
    return image.rotate(random.uniform(-30, 30))

def random_flip(image):
    if random.random() > 0.5:
        return ImageOps.mirror(image)
    return image

def random_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(0.5, 1.5))

def random_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(random.uniform(0.5, 1.5))

def random_augmentation(image):
    augmentations = [random_rotation, random_flip, random_brightness, random_contrast]
    augmentation = random.choice(augmentations)
    return augmentation(image)

# Augment class images to balance dataset
def augment_class(class_dir, target_count):
    class_images = os.listdir(class_dir)
    current_count = len(class_images)
    augmented_images = []

    while current_count < target_count:
        img_name = random.choice(class_images)
        img_path = os.path.join(class_dir, img_name)
        image = Image.open(img_path)
        augmented_image = random_augmentation(image)
        augmented_images.append(augmented_image)
        current_count += 1

    return augmented_images

for class_name in class_dirs:
    class_dir = os.path.join(dataset_dir, class_name)
    target_dir = os.path.join(output_dir, class_name)
    os.makedirs(target_dir, exist_ok=True)

    for img_name in os.listdir(class_dir):
        shutil.copy(os.path.join(class_dir, img_name), target_dir)

    augmented_images = augment_class(class_dir, max_count)
    for idx, augmented_image in enumerate(augmented_images):
        augmented_image.save(os.path.join(target_dir, f'aug_{idx}.jpg'))

balanced_dataset_dir = output_dir
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    balanced_dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    balanced_dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# BUILD MODEL
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in mobilenet.layers:
    layer.trainable = False

last_layer = mobilenet.get_layer('Conv_1')
last_output = last_layer.output

x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(4, activation='softmax')(x)

model = Model(inputs=mobilenet.input, outputs=prediction)

model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
checkpoint = ModelCheckpoint('hairapy_model.h5', monitor='val_loss', save_best_only=True, mode='min')

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[early_stopping, lr_scheduler, checkpoint]
)

# Save the model in SavedModel format
saved_model_dir = 'saved_model/hairapy_model'
tf.saved_model.save(model, saved_model_dir)

# Convert to TensorFlow.js format
os.system('tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model saved_model/hairapy_model tfjs_model')