import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path
from tensorflow.keras.preprocessing import image
from Transformation import Transformation

# Configuration (Doit être identique à train.py)
IMG_HEIGHT = 256
IMG_WIDTH = 256
MODEL_PATH = "dataset_and_model/leaf_model.keras"

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
    
    # Extraire la classe réelle (nom du dossier parent)
    true_class = Path(img_path).parent.name

    # 2. Chargement du modèle
    if not os.path.exists(MODEL_PATH):
        msg = (
            f"Erreur : Le modèle '{MODEL_PATH}' est introuvable. "
            "Lancez train.py d'abord."
        )
        print(msg)
        return

    print("Chargement du modèle...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    # 3. Prétraitement de l'image (Transformation)
    # Charger l'image originale avec OpenCV
    img_original_cv = cv.imread(img_path)
    if img_original_cv is None:
        print(f"Erreur : Impossible de charger l'image '{img_path}'.")
        return

    # Appliquer la transformation (comme pendant l'entraînement)
    transformer = Transformation(img_original_cv)
    img_masked = transformer.masked_leaf()

    # Redimensionner l'image transformée
    img_transformed = cv.resize(
        img_masked, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA
    )

    # IMPORTANT: OpenCV charge en BGR, TensorFlow attend RGB
    img_rgb = cv.cvtColor(img_transformed, cv.COLOR_BGR2RGB)

    # Conversion en tableau numpy pour le modèle
    img_array = img_rgb.astype(np.float32)

    # Création d'un batch
    # Le modèle attend un tableau d'images, pas une seule image
    # On passe de (256, 256, 3) à (1, 256, 256, 3)
    img_batch = tf.expand_dims(img_array, 0)

    # 4. Prédiction
    print("Analyse de l'image...")
    predictions = model.predict(img_batch)

    # On prend l'index de la probabilité la plus élevée
    # Transforme en pourcentages
    score = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(score)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = 100 * np.max(score)

    print("\n------------------------------------------------")
    print(f"True class: {true_class}")
    print(f"Class predicted: {predicted_class_name}")
    print(f"Confiance : {confidence:.2f}%")
    print("------------------------------------------------")

    # 5. Affichage (Bonus visuel demandé par le sujet)
    # On charge l'image originale pour l'affichage
    img_original = image.load_img(img_path)

    plt.figure(figsize=(10, 5))

    # Image Originale
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image\nTrue Class: {true_class}")
    plt.imshow(img_original)
    plt.axis("off")

    # Image Transformée (celle vue par le réseau)
    plt.subplot(1, 2, 2)
    plt.title(f"Transformed (256x256)\nPredicted: {predicted_class_name}")
    # Convertir BGR vers RGB pour matplotlib
    img_transformed_rgb = cv.cvtColor(img_transformed, cv.COLOR_BGR2RGB)
    plt.imshow(img_transformed_rgb)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
