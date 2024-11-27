import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog

# Path to the dataset directory
data_dir = "c:\\Users\\HP\\Desktop\\Lungs detection dataset\\LungCancer_Dataset-main"




# Image size to resize (standardize)
img_size = (128, 128)

# Function to load images and extract features
def load_images_and_labels(data_dir, img_size):
    features = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)

                # Extract HOG features
                hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                   block_norm='L2-Hys', visualize=False)

                features.append(hog_features)
                labels.append(label)

    return np.array(features), np.array(labels)

# Load the images and extract features
X, y = load_images_and_labels(data_dir, img_size)

from sklearn.preprocessing import LabelEncoder

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Create a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display feature importance
feature_importances = rf_model.feature_importances_
print(f"Top 10 feature importances: {np.argsort(feature_importances)[-10:][::-1]}")

import matplotlib.pyplot as plt
def show_sample_images(data_dir, img_size, num_samples=5):
    # Create a figure with the number of subplots equal to num_samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    sample_count = 0

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)

        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue

        # Iterate through images in the directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            img = cv2.resize(img, img_size)
            axes[sample_count].imshow(img, cmap='gray')
            axes[sample_count].set_title(f'Label: {label}')
            axes[sample_count].axis('off')
            sample_count += 1

            # Stop if we have reached the required number of samples
            if sample_count >= num_samples:
                break

        if sample_count >= num_samples:
            break

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Function to visualize HOG features
def visualize_hog(img, img_size):
    img_resized = cv2.resize(img, img_size)
    hog_features, hog_image = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  block_norm='L2-Hys', visualize=True, channel_axis=-1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.show()

# Function to plot feature importance
def plot_feature_importance(rf_model):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:][::-1]  # Top 10 features
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], color="y", align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, le):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Example Usage
# Replace 'data_dir' and 'img_size' with your actual directory path and image size.
data_dir = "c:\\Users\\HP\\Desktop\\Lungs detection dataset\\LungCancer_Dataset-main"

img_size = (128, 128)  # Example size, adjust as needed

# Show some sample images
show_sample_images(data_dir, img_size)

# Load a sample image directly
sample_img_path = os.path.join(data_dir, 'PositiveSample.PNG')
sample_img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)

if sample_img is not None:
    # Visualize HOG features for the sample image
    visualize_hog(sample_img, img_size)
else:
    print(f"Error: Could not read the sample image at {sample_img_path}")
