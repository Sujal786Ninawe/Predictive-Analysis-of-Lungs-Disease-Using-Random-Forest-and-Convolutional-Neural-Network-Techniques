import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.camera import Camera
from kivy.uix.screenmanager import ScreenManager, Screen
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
import os

# Load both models (adjust the paths to your model files)
LUNG_CANCER_MODEL_PATH = 'C:/Users/HP/Desktop/Lungs detection dataset/lung_cancer_model.h5'
PRETRAINED_MODEL_PATH = 'lung_disease_detection_model.h5'

class MainScreen(Screen):
    pass

class CameraScreen(Screen):
    pass

class LungDiseaseApp(App):
    def build(self):
        Window.size = (400, 600)  # Set window size for better visualization during development

        # Load your machine learning models
        self.lung_cancer_model = load_model(LUNG_CANCER_MODEL_PATH)
        self.pretrained_model = load_model(PRETRAINED_MODEL_PATH)

        # Set up screen manager
        self.sm = ScreenManager()

        # Main screen for model selection
        self.main_screen = MainScreen(name='main')
        self.setup_main_screen()
        self.sm.add_widget(self.main_screen)

        # Camera screen for capturing image
        self.camera_screen = CameraScreen(name='camera')
        self.setup_camera_screen()
        self.sm.add_widget(self.camera_screen)

        return self.sm

    def setup_main_screen(self):
        # Main layout
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Label for instructions
        label = Label(text="Choose Detection Type", font_size=18)
        layout.add_widget(label)

        # Buttons for selecting detection type
        lung_cancer_button = Button(text="Detect Lung Cancer", font_size=16)
        lung_cancer_button.bind(on_press=self.detect_lung_cancer)
        layout.add_widget(lung_cancer_button)

        pretrained_button = Button(text="Detect Pneumonia, Tuberculosis, COVID-19, Normal", font_size=16)
        pretrained_button.bind(on_press=self.detect_pretrained_diseases)
        layout.add_widget(pretrained_button)

        # Button to switch to camera mode
        camera_button = Button(text="Open Camera", font_size=16)
        camera_button.bind(on_press=self.switch_to_camera)
        layout.add_widget(camera_button)

        # Placeholder for displaying the image
        self.image_display = Image(size_hint=(1, 0.6))
        layout.add_widget(self.image_display)

        # Result label
        self.result_label = Label(text="", font_size=18)
        layout.add_widget(self.result_label)

        self.main_screen.add_widget(layout)

    def setup_camera_screen(self):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Add Kivy's Camera widget
        self.camera = Camera(play=True, resolution=(640, 480))
        layout.add_widget(self.camera)

        # Button to capture an image
        capture_button = Button(text="Capture Image", size_hint=(1, 0.2))
        capture_button.bind(on_press=self.capture_image)
        layout.add_widget(capture_button)

        self.camera_screen.add_widget(layout)

    def switch_to_camera(self, instance):
        # Switch to the camera screen
        self.sm.current = 'camera'

    def switch_to_main(self):
        # Switch back to the main screen
        self.sm.current = 'main'

    def capture_image(self, instance):
        # Capture the current frame from the camera and save it as an image
        texture = self.camera.texture
        image_data = texture.pixels
        size = texture.size

        # Create a PIL image from the texture data
        pil_img = PILImage.frombytes(mode='RGBA', size=size, data=image_data)
        pil_img = pil_img.convert('RGB')  # Convert to RGB format
        image_path = 'captured_image.jpg'
        pil_img.save(image_path)

        # Switch back to the main screen
        self.switch_to_main()

        # Display the captured image
        self.load_image([image_path])

    def detect_lung_cancer(self, instance):
        # Set the model type and open the file chooser
        self.model_type = 'lung_cancer'
        self.show_file_chooser()

    def detect_pretrained_diseases(self, instance):
        # Set the model type and open the file chooser
        self.model_type = 'pretrained'
        self.show_file_chooser()

    def show_file_chooser(self):
        # File chooser to upload images
        file_chooser_layout = BoxLayout(orientation='vertical', spacing=10)
        
        file_chooser = FileChooserIconView(filters=["*.png", "*.jpg", "*.jpeg"])
        file_chooser_layout.add_widget(file_chooser)
        
        # Button to confirm image selection
        select_button = Button(text="Select Image", size_hint=(1, 0.2))
        select_button.bind(on_press=lambda x: self.load_image(file_chooser.selection))
        file_chooser_layout.add_widget(select_button)
        
        # Popup for file chooser
        popup = Popup(title='Choose an image', content=file_chooser_layout, size_hint=(0.9, 0.9))
        self.file_chooser_popup = popup
        popup.open()

    def load_image(self, selection):
        if selection:
            # Get the selected image file path
            image_path = selection[0]
            self.file_chooser_popup.dismiss()

            # Display the selected image
            pil_img = PILImage.open(image_path)
            pil_img = pil_img.convert('RGB').resize((400, 400))  # Resize for display
            
            # Convert to texture for Kivy Image widget
            texture = Texture.create(size=(pil_img.width, pil_img.height))
            texture.blit_buffer(np.array(pil_img).tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            self.image_display.texture = texture

            # Predict the disease based on the selected model
            if self.model_type == 'lung_cancer':
                self.predict_lung_cancer(image_path)
            else:
                self.predict_pretrained_diseases(image_path)

    def predict_lung_cancer(self, image_path):
        # Preprocess the image for lung cancer model
        img = PILImage.open(image_path).convert('RGB').resize((224, 224))  # Ensure this matches your model's input size
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction using lung cancer model
        predictions = self.lung_cancer_model.predict(img_array)
        classes = ['Lung Cancer Negative', 'Lung Cancer Positive']
        predicted_class = classes[np.argmax(predictions)]

        # Display the prediction result
        self.result_label.text = f"Prediction: {predicted_class}"

    def predict_pretrained_diseases(self, image_path):
        # Preprocess the image for pretrained model
        img = PILImage.open(image_path).convert('RGB').resize((224, 224))  # Ensure this matches your model's input size
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction using pretrained model
        predictions = self.pretrained_model.predict(img_array)
        classes = ['Normal', 'Pneumonia', 'Tuberculosis', 'COVID-19']
        predicted_class = classes[np.argmax(predictions)]

        # Display the prediction result
        self.result_label.text = f"Prediction: {predicted_class}"


if __name__ == '__main__':
    LungDiseaseApp().run()
