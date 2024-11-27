# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

# Corrected imports using tensorflow.keras
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Resize the image
IMAGE_SIZE = [224, 224]
categories = ["Negative", "Positive"]

# File path to all required folders
dataset_path = 'C:/Users/HP/Desktop/Lungs detection dataset/LungCancer_Dataset-main'

train_negative_images = glob(dataset_path + "/Negative/*")
train_positive_images = glob(dataset_path + "/Positive/*")

plt.figure(figsize=(10, 8), dpi=80)
init_subplot = 230
for i in range(1, 7):
    plt.subplot(init_subplot + i)

    if i < 4:
        img = Image.open(np.random.choice(train_negative_images)).resize((224, 224))  # Corrected to 224x224
        plt.title("Negative Image")
    else:
        img = Image.open(np.random.choice(train_positive_images)).resize((224, 224))  # Corrected to 224x224
        plt.title("Positive Image")

    img = np.asarray(img)
    plt.axis('off')
    plt.imshow(img)

plt.show()

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape = IMAGE_SIZE + [3],
            weights = 'imagenet',
            include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(1000, activation='relu')(x)

# This is the Last Layer with softmax activation for binary outputs
prediction = Dense(len(categories), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(dataset_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(dataset_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

print(f"Validation data size: {len(test_set)}")
print(f"Batch size: {test_set.batch_size}")

r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set) // training_set.batch_size,
  validation_steps=len(test_set) // test_set.batch_size
)

fig, ax = plt.subplots(figsize=(20, 4), nrows=1, ncols=2)

ax[0].plot(r.history['loss'], color='b', label="Training Loss")
ax[0].plot(r.history['val_loss'], color='r', label="Testing Loss")  # Removed `axes=ax[0]`
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(r.history['accuracy'], color='b', label="Training Accuracy")
ax[1].plot(r.history['val_accuracy'], color='r', label="Testing Accuracy")
legend = ax[1].legend(loc='best', shadow=True)


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Use tensorflow.keras instead of keras

# Specify the path to your image
path = "C:/Users/HP\Desktop/Lungs detection dataset/LungCancer_Dataset-main/Positive/person11_bacteria_45.jpeg"

# Load and preprocess the image
img = load_img(path, target_size=(224, 224))  # Resize to match model input size
input_arr = img_to_array(img) / 255.0  # Normalize pixel values

# Display the image
plt.imshow(input_arr)
plt.axis('off')  # Turn off axis labels
plt.show()

# Print the shape of the input array
print("Input array shape (before expanding dimensions):", input_arr.shape)

# Expand dimensions to match the input shape expected by the model (batch size, height, width, channels)
input_arr = np.expand_dims(input_arr, axis=0)
print("Input array shape (after expanding dimensions):", input_arr.shape)

# Assuming `model` is already loaded, replace this with your model loading if necessary
# model = load_model('path_to_your_model.h5')  # Uncomment this line if you need to load the model

# Get the prediction (ensure `model` is defined and loaded)
pred = model.predict(input_arr)
pred_class = np.argmax(pred, axis=1)[0]  # Get the class with the highest probability

# Print the prediction
if pred_class == 0:  # Assuming class 0 is 'Negative'
    print("The patient is safe or not under cancer")
else:  # Assuming class 1 is 'Positive'
    print("The patient is in danger or having cancer")

# Assuming 'model' is your trained model
model.save('C:/Users/HP/Desktop/Lungs detection dataset/lung_cancer_model.h5')
