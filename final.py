import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Load the pretrained model
plant_species_model = tf.keras.models.load_model('models/plantSpecies.h5')

# Load and preprocess the image
img_path = "plant_images/Apple___Apple_scab/0b1e31fa-cbc0-41ed-9139-c794e6855e82___FREC_Scab 3089_90deg.JPG"
target_size = (256, 256)  # Adjust target_size as needed

img = image.load_img(img_path, target_size=target_size)
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0  # Normalize pixel values (if required)

# Make predictions
predictions = plant_species_model.predict(img)

# Define class labels (replace with your class labels)
species_class_labels = ['apple','cherry','grape','peach','pepper bell']

# Get the predicted class
predicted_species = species_class_labels[np.argmax(predictions)]

if ( predicted_species == "apple" ):
    disease_model = tf.keras.models.load_model('models/apple.h5')
    disease_class_labels = ['apple scarb', 'black rot', 'cedar apple rust', 'healthy']

elif ( predicted_species == "cherry" ):
    disease_model = tf.keras.models.load_model('models/cherry.h5')
    disease_class_labels = ['powdery mildew','healthy']

elif ( predicted_species == "grape" ):
    disease_model = tf.keras.models.load_model('models/grapes.h5')
    disease_class_labels = ['black rot', 'esca','leaf blight','healthy']

elif ( predicted_species == "peach" ):
    disease_model = tf.keras.models.load_model('models/peach(1).h5')
    disease_class_labels = ['bacterial spot','healthy']

elif ( predicted_species == "pepper bell" ):
    disease_model = tf.keras.models.load_model('models/pepper.h5')
    disease_class_labels = ['bacterial spot','healthy']

predictions = disease_model.predict(img)
predicted_disease = disease_class_labels[np.argmax(predictions)]

# Display the predicted class
print(f"Disease: {predicted_disease} Plant: {predicted_species}")
