import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import random
from keras.preprocessing.image import load_img, img_to_array

# Directories for the dataset
train_data_dir = 'C:\\Users\\HP\\Desktop\\Lungs detection dataset\\LungXRays-grayscale\\train'
val_data_dir = 'C:\\Users\\HP\\Desktop\\Lungs detection dataset\\LungXRays-grayscale\\val'
test_data_dir = 'C:\\Users\\HP\\Desktop\\Lungs detection dataset\\LungXRays-grayscale\\test'

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generators for validation and testing data (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading training, validation, and test data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Base directory containing train, test, and val folders
base_dir = 'C:\\Users\\HP\\Desktop\\Lungs detection dataset\\LungXRays-grayscale'

# Categories and subdirectories
categories = ['Normal', 'Corona Virus Disease', 'Pneumonia', 'Tuberculosis']
folders = ['train', 'test', 'val']

def get_random_image_path(folder, category):
    """
    Get a random image path from a specified folder and category.
    """
    category_dir = os.path.join(base_dir, folder, category)
    if not os.path.exists(category_dir):
        raise FileNotFoundError(f"The directory {category_dir} does not exist.")

    image_name = random.choice(os.listdir(category_dir))
    return os.path.join(category_dir, image_name)

def display_images():
    plt.figure(figsize=(15, 10))

    for i, category in enumerate(categories, 1):
        for j, folder in enumerate(folders):
            image_path = get_random_image_path(folder, category)
            try:
                img = load_img(image_path, target_size=(128, 128), color_mode="grayscale")
                img_array = img_to_array(img) / 255.0
                plt.subplot(len(categories), len(folders), i + j * len(categories))
                plt.imshow(img_array.squeeze(), cmap='gray')
                plt.title(f'{category} ({folder})')
                plt.axis('off')

            except Exception as e:
                print(f"Error loading image for {category} in {folder}: {e}")

    plt.tight_layout()
    plt.show()

# Display sample images
display_images()

# Model building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes: Normal, Corona Virus Disease, Pneumonia, Tuberculosis
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=50
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc:.2f}')

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Save the trained model
model.save('lung_disease_detection_model.h5')
