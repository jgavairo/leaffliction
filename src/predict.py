import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Configuration (Doit être identique à train.py)
IMG_HEIGHT = 256
IMG_WIDTH = 256
MODEL_PATH = "leaf_model.keras"

# TRES IMPORTANT : L'ordre alphabétique exact donné par train.py
CLASS_NAMES = [
    'Apple_Black_rot', 
    'Apple_healthy', 
    'Apple_rust', 
    'Apple_scab', 
    'Grape_Black_rot', 
    'Grape_Esca', 
    'Grape_healthy', 
    'Grape_spot'
]

def main():
    # 1. Vérification des arguments
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <path_to_image>")
        return

    img_path = sys.argv[1]
    
    if not os.path.exists(img_path):
        print(f"Erreur : L'image '{img_path}' n'existe pas.")
        return

    # 2. Chargement du modèle
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Le modèle '{MODEL_PATH}' est introuvable. Lancez train.py d'abord.")
        return
        
    print("Chargement du modèle...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    # 3. Prétraitement de l'image (Transformation)
    # On charge l'image en la redimensionnant directement (comme lors de l'entraînement)
    # C'est l'image "Transformée"
    img_transformed = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    
    # Conversion en tableau numpy
    img_array = image.img_to_array(img_transformed)
    
    # Création d'un batch (le modèle attend un tableau d'images, pas une seule image)
    # On passe de (256, 256, 3) à (1, 256, 256, 3)
    img_batch = tf.expand_dims(img_array, 0)

    # 4. Prédiction
    print("Analyse de l'image...")
    predictions = model.predict(img_batch)
    
    # On prend l'index de la probabilité la plus élevée
    score = tf.nn.softmax(predictions[0]) # Transforme en pourcentages
    predicted_class_index = np.argmax(score)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = 100 * np.max(score)

    print(f"\n------------------------------------------------")
    print(f"Class predicted: {predicted_class_name}")
    print(f"Confiance : {confidence:.2f}%")
    print(f"------------------------------------------------")

    # 5. Affichage (Bonus visuel demandé par le sujet)
    # On charge l'image originale pour l'affichage
    img_original = image.load_img(img_path)

    plt.figure(figsize=(10, 5))
    
    # Image Originale
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_original)
    plt.axis("off")

    # Image Transformée (celle vue par le réseau)
    # Note: On convertit en uint8 pour l'affichage correct par matplotlib
    plt.subplot(1, 2, 2)
    plt.title(f"Transformed (256x256)\nPred: {predicted_class_name}")
    plt.imshow(img_transformed)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()