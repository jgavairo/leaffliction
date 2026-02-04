import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
from collections import defaultdict

# Configuration (Doit être identique à train.py et predict.py)
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

def predict_image(model, img_path):
    """
    Prédit la classe d'une image
    Retourne: (predicted_class_name, confidence, predicted_class_index)
    """
    # Chargement et prétraitement de l'image
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_batch = tf.expand_dims(img_array, 0)
    
    # Prédiction
    predictions = model.predict(img_batch, verbose=0)
    score = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(score)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = 100 * np.max(score)
    
    return predicted_class_name, confidence, predicted_class_index

def test_directory(model, data_dir, max_images_per_class=None):
    """
    Test le modèle sur toutes les images du répertoire
    et calcule l'accuracy pour chaque classe
    
    Args:
        model: Le modèle chargé
        data_dir: Le répertoire contenant les sous-dossiers de classes
        max_images_per_class: Nombre max d'images à tester par classe (None = toutes)
    
    Returns:
        dict avec les statistiques par classe
    """
    # Statistiques par classe
    stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'accuracy': 0.0,
        'predictions': defaultdict(int)  # Compte les prédictions vers chaque classe
    })
    
    # Parcourir chaque classe (sous-dossier)
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Erreur : Le dossier '{data_dir}' n'existe pas.")
        return None
    
    # Liste des dossiers de classes
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"Erreur : Aucun sous-dossier trouvé dans '{data_dir}'.")
        return None
    
    print(f"\nTest du modèle sur : {data_dir}")
    print(f"Classes trouvées : {[d.name for d in class_dirs]}")
    print("=" * 80)
    
    # Pour chaque classe
    for class_dir in sorted(class_dirs):
        true_class = class_dir.name
        
        # Lister toutes les images
        image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(class_dir.glob(ext))
        
        # Limiter le nombre d'images si demandé
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        num_images = len(image_files)
        
        if num_images == 0:
            print(f"\n⚠️  Classe '{true_class}': Aucune image trouvée, ignorée.")
            continue
        
        print(f"\n📁 Classe '{true_class}': {num_images} images à tester...")
        
        # Tester chaque image
        for idx, img_path in enumerate(image_files, 1):
            try:
                predicted_class, confidence, _ = predict_image(model, str(img_path))
                
                # Mise à jour des statistiques
                stats[true_class]['total'] += 1
                stats[true_class]['predictions'][predicted_class] += 1
                
                if predicted_class == true_class:
                    stats[true_class]['correct'] += 1
                else:
                    stats[true_class]['incorrect'] += 1
                
                # Affichage de la progression (tous les 10%)
                if idx % max(1, num_images // 10) == 0 or idx == num_images:
                    progress = (idx / num_images) * 100
                    current_acc = (stats[true_class]['correct'] / stats[true_class]['total']) * 100
                    print(f"  Progression: {idx}/{num_images} ({progress:.0f}%) - Accuracy actuelle: {current_acc:.2f}%")
                    
            except Exception as e:
                print(f"  ⚠️  Erreur avec {img_path.name}: {e}")
    
    # Calculer l'accuracy finale pour chaque classe
    for class_name in stats:
        if stats[class_name]['total'] > 0:
            stats[class_name]['accuracy'] = (stats[class_name]['correct'] / stats[class_name]['total']) * 100
    
    return dict(stats)

def print_results(stats):
    """
    Affiche les résultats de manière formatée
    """
    print("\n" + "=" * 80)
    print("RÉSULTATS DU TEST".center(80))
    print("=" * 80)
    
    total_images = 0
    total_correct = 0
    
    # Résultats par classe
    for class_name in sorted(stats.keys()):
        s = stats[class_name]
        total_images += s['total']
        total_correct += s['correct']
        
        print(f"\n📊 Classe: {class_name}")
        print(f"   Total d'images testées: {s['total']}")
        print(f"   Prédictions correctes:  {s['correct']}")
        print(f"   Prédictions incorrectes: {s['incorrect']}")
        print(f"   ✅ ACCURACY: {s['accuracy']:.2f}%")
        
        # Afficher la matrice de confusion simplifiée pour cette classe
        if s['incorrect'] > 0:
            print(f"   Confusions:")
            for pred_class, count in sorted(s['predictions'].items()):
                if pred_class != class_name and count > 0:
                    print(f"      → {pred_class}: {count} fois")
    
    # Accuracy globale
    print("\n" + "=" * 80)
    if total_images > 0:
        global_accuracy = (total_correct / total_images) * 100
        print(f"🎯 ACCURACY GLOBALE: {global_accuracy:.2f}%")
        print(f"   ({total_correct}/{total_images} prédictions correctes)")
    print("=" * 80)

def main():
    # 1. Vérification des arguments
    if len(sys.argv) < 2:
        print("Usage: python src/test_accuracy.py <path_to_data_dir> [max_images_per_class]")
        print("\nExemples:")
        print("  python src/test_accuracy.py output/augmented_data")
        print("  python src/test_accuracy.py output/augmented_data 50")
        return
    
    data_dir = sys.argv[1]
    max_images = None
    
    if len(sys.argv) >= 3:
        try:
            max_images = int(sys.argv[2])
            print(f"Limite: {max_images} images par classe")
        except ValueError:
            print("Attention: Le paramètre max_images doit être un nombre entier.")
            return
    
    # 2. Chargement du modèle
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Le modèle '{MODEL_PATH}' est introuvable.")
        print("Veuillez d'abord entraîner le modèle avec train.py")
        return
    
    print("Chargement du modèle...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modèle chargé avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return
    
    # 3. Test du modèle
    stats = test_directory(model, data_dir, max_images)
    
    if stats is None:
        return
    
    # 4. Affichage des résultats
    print_results(stats)

if __name__ == "__main__":
    main()
