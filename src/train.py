import sys
import os
import tensorflow as tf
import shutil
import hashlib
import csv
import cv2 as cv
import random
import re
from pathlib import Path
from collections import defaultdict
from Transformation import Transformation
from augment import augment_class

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32

def extract_base_name(filename):
    """
    Extrait le base name d'un fichier en supprimant les suffixes d'augmentation.
    Ex: 'image (1) Flip.JPG' -> 'image (1)'
        'image (1).JPG' -> 'image (1)'
    """
    # Patterns d'augmentation courants
    augmentation_patterns = [
        r'\s+Flip', r'\s+Rotate', r'\s+Skew', r'\s+Brightness',
        r'\s+Contrast', r'\s+Blur', r'\s+Noise', r'\s+Scale',
        r'\s+Crop', r'\s+Zoom', r'\s+Shift', r'\s+Perspective',
        r'_augmented_\d+', r'_aug\d+', r'augmented_\d+_'
    ]
    
    stem = Path(filename).stem
    
    # Essayer de retirer les patterns d'augmentation
    for pattern in augmentation_patterns:
        stem = re.sub(pattern, '', stem, flags=re.IGNORECASE)
    
    return stem.strip()

def group_images_by_base(class_dir):
    """
    Groupe les images par leur base name (image originale + ses augmentations).
    
    Returns: dict {base_name: [list of image paths]}
    """
    img_extensions = {".jpg", ".jpeg", ".png"}
    images = [f for f in class_dir.iterdir() if f.suffix.lower() in img_extensions]
    
    groups = defaultdict(list)
    for img_path in images:
        base_name = extract_base_name(img_path.name)
        groups[base_name].append(img_path)
    
    return groups

def split_by_groups(class_dir, train_ratio=0.8, seed=123):
    """
    Split les images en train/val en respectant les groupes (image mère + augmentations).
    
    Returns: {'train': [img_paths], 'val': [img_paths]}
    """
    random.seed(seed)
    
    # Grouper par base name
    groups = group_images_by_base(class_dir)
    
    # Mélanger les groupes (pas les images individuelles)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    
    # Calculer le split
    train_count = int(len(group_keys) * train_ratio)
    
    train_groups = group_keys[:train_count]
    val_groups = group_keys[train_count:]
    
    # Collecter toutes les images de chaque groupe
    train_images = []
    val_images = []
    
    for group_key in train_groups:
        train_images.extend(groups[group_key])
    
    for group_key in val_groups:
        val_images.extend(groups[group_key])
    
    return {'train': train_images, 'val': val_images}

def main():
    print("Script démarré...")
    print(f"Arguments: {sys.argv}")

    # 1. Vérification des arguments
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_dataset>")
        print("Example: python train.py ./Apple/")
        return

    data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Erreur : Le dossier '{data_dir}' n'existe pas.")
        return
    
    print(f"Chargement des images depuis : {data_dir}")

    # 2. Split AVANT transformation pour éviter data leakage
    print("\n--- Split Train/Validation (80/20) par groupes ---")
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"Erreur : Aucun sous-dossier trouvé dans '{data_dir}'")
        return
    
    split_data = {}
    for class_dir in subdirs:
        split_data[class_dir.name] = split_by_groups(class_dir, train_ratio=0.8, seed=123)
        print(f"{class_dir.name}: {len(split_data[class_dir.name]['train'])} train, {len(split_data[class_dir.name]['val'])} val")

    # 3. Création des dossiers de sortie
    output_base = Path("temp_training_data")
    
    # Nettoyage si existe déjà
    if output_base.exists():
        shutil.rmtree(output_base)
    
    output_train = output_base / "train"
    output_val = output_base / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png"}
    
    # 4. Transformation des images TRAIN
    print("\n--- Transformation des images TRAIN ---")
    train_features_csv = output_train / "features.csv"
    csv_fieldnames = [
        "file", "mask_path", "norm_path", "area", "perimeter",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "mean_r", "mean_g", "mean_b"
    ]
    
    with open(train_features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
    
    train_transformed = output_train / "transformed"
    train_transformed.mkdir(parents=True, exist_ok=True)
    
    total_train = 0
    for class_name, splits in split_data.items():
        print(f"Transformation TRAIN de {class_name}...")
        for img_path in splits['train']:
            try:
                image = cv.imread(str(img_path))
                if image is None:
                    continue
                
                transformer = Transformation(image)
                
                # Préparer les dossiers de sortie
                masks_out = train_transformed / "masks" / class_name
                norm_out = train_transformed / "normalized" / class_name
                masks_out.mkdir(parents=True, exist_ok=True)
                norm_out.mkdir(parents=True, exist_ok=True)
                
                # Générer masque et image normalisée
                masked = transformer.masked_leaf()
                mask = transformer.mask
                
                # Calculer features
                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                area = 0
                perimeter = 0
                bbox = (0, 0, 0, 0)
                if contours:
                    largest = max(contours, key=cv.contourArea)
                    area = float(cv.contourArea(largest))
                    perimeter = float(cv.arcLength(largest, True))
                    x, y, w, h = cv.boundingRect(largest)
                    bbox = (int(x), int(y), int(w), int(h))
                
                mean_bgr = cv.mean(image, mask=mask)[:3]
                mean_r, mean_g, mean_b = float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])
                
                # Sauvegarder fichiers
                base_name = img_path.stem
                ext = img_path.suffix
                mask_name = f"{base_name}_mask.png"
                norm_name = f"{base_name}_norm{ext}"
                
                cv.imwrite(str(masks_out / mask_name), mask)
                
                # Image normalisée redimensionnée
                norm = cv.resize(masked, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
                cv.imwrite(str(norm_out / norm_name), norm)
                
                # Ajouter features au CSV
                features = {
                    "file": str(img_path),
                    "mask_path": str(masks_out / mask_name),
                    "norm_path": str(norm_out / norm_name),
                    "area": area,
                    "perimeter": perimeter,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "mean_r": mean_r,
                    "mean_g": mean_g,
                    "mean_b": mean_b,
                }
                
                with open(train_features_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    writer.writerow(features)
                
                total_train += 1
                
            except Exception as e:
                print(f"  Erreur avec {img_path.name}: {e}")
    
    print(f"Total d'images TRAIN transformées : {total_train}")

    # 5. Transformation des images VALIDATION
    print("\n--- Transformation des images VALIDATION ---")
    val_features_csv = output_val / "features.csv"
    
    with open(val_features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
    
    val_transformed = output_val / "transformed"
    val_transformed.mkdir(parents=True, exist_ok=True)
    
    total_val = 0
    for class_name, splits in split_data.items():
        print(f"Transformation VALIDATION de {class_name}...")
        for img_path in splits['val']:
            try:
                image = cv.imread(str(img_path))
                if image is None:
                    continue
                
                transformer = Transformation(image)
                
                masks_out = val_transformed / "masks" / class_name
                norm_out = val_transformed / "normalized" / class_name
                masks_out.mkdir(parents=True, exist_ok=True)
                norm_out.mkdir(parents=True, exist_ok=True)
                
                masked = transformer.masked_leaf()
                mask = transformer.mask
                
                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                area = 0
                perimeter = 0
                bbox = (0, 0, 0, 0)
                if contours:
                    largest = max(contours, key=cv.contourArea)
                    area = float(cv.contourArea(largest))
                    perimeter = float(cv.arcLength(largest, True))
                    x, y, w, h = cv.boundingRect(largest)
                    bbox = (int(x), int(y), int(w), int(h))
                
                mean_bgr = cv.mean(image, mask=mask)[:3]
                mean_r, mean_g, mean_b = float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])
                
                base_name = img_path.stem
                ext = img_path.suffix
                mask_name = f"{base_name}_mask.png"
                norm_name = f"{base_name}_norm{ext}"
                
                cv.imwrite(str(masks_out / mask_name), mask)
                
                norm = cv.resize(masked, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
                cv.imwrite(str(norm_out / norm_name), norm)
                
                features = {
                    "file": str(img_path),
                    "mask_path": str(masks_out / mask_name),
                    "norm_path": str(norm_out / norm_name),
                    "area": area,
                    "perimeter": perimeter,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "mean_r": mean_r,
                    "mean_g": mean_g,
                    "mean_b": mean_b,
                }
                
                with open(val_features_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    writer.writerow(features)
                
                total_val += 1
                
            except Exception as e:
                print(f"  Erreur avec {img_path.name}: {e}")
    
    print(f"Total d'images VALIDATION transformées : {total_val}")

    # 6. Augmentation supplémentaire des images TRAIN uniquement
    print("\n--- Augmentation supplémentaire des images TRAIN ---")
    train_normalized = train_transformed / "normalized"
    train_augmented = output_train / "augmented"
    train_augmented.mkdir(parents=True, exist_ok=True)
    
    # Compter les images par classe dans TRAIN
    class_counts = {}
    for class_dir in train_normalized.iterdir():
        if class_dir.is_dir():
            img_count = len([f for f in class_dir.iterdir() if f.suffix.lower() in img_extensions])
            class_counts[class_dir] = img_count
    
    # Équilibrer les classes
    if class_counts:
        max_count = max(class_counts.values())
        print(f"Nombre maximum d'images par classe (TRAIN) : {max_count}")
        
        for class_dir, count in class_counts.items():
            deficit = max_count - count
            if deficit > 0:
                print(f"Augmentation de {class_dir.name} : {count} -> {max_count} (+{deficit} images)")
                augment_class(class_dir, deficit, train_augmented, verbose=True)
            else:
                print(f"Pas d'augmentation nécessaire pour {class_dir.name}")
    
    # 7. Fusionner transformées + augmentées pour TRAIN
    train_final = output_train / "final"
    train_final.mkdir(parents=True, exist_ok=True)
    
    # Copier images transformées TRAIN
    for class_dir in train_normalized.iterdir():
        if class_dir.is_dir():
            dest_class = train_final / class_dir.name
            dest_class.mkdir(parents=True, exist_ok=True)
            for img in class_dir.iterdir():
                if img.suffix.lower() in img_extensions:
                    shutil.copy(img, dest_class / img.name)
    
    # Copier images augmentées TRAIN
    for class_dir in train_augmented.iterdir():
        if class_dir.is_dir():
            dest_class = train_final / class_dir.name
            dest_class.mkdir(parents=True, exist_ok=True)
            for img in class_dir.iterdir():
                if img.suffix.lower() in img_extensions:
                    shutil.copy(img, dest_class / img.name)
    
    # 8. Préparer le dataset VALIDATION (sans augmentation supplémentaire)
    val_final = output_val / "final"
    val_final.mkdir(parents=True, exist_ok=True)
    
    val_normalized = val_transformed / "normalized"
    for class_dir in val_normalized.iterdir():
        if class_dir.is_dir():
            dest_class = val_final / class_dir.name
            dest_class.mkdir(parents=True, exist_ok=True)
            for img in class_dir.iterdir():
                if img.suffix.lower() in img_extensions:
                    shutil.copy(img, dest_class / img.name)
    
    # 9. Chargement des datasets (pas de split, déjà séparés)
    print(f"\n--- Préparation des datasets ---")
    print(f"Train: {train_final}")
    print(f"Validation: {val_final}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_final),
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(val_final),
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 10. Vérification
    class_names = train_ds.class_names
    print(f"\nClasses trouvées : {class_names}")
    print(f"Nombre de classes : {len(class_names)}")

    # 8. Optimisation des performances
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    # 9. Construction du modèle CNN
    print("\n--- Construction du modèle CNN ---")
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        # Bloc 1
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
            
        # Bloc 2
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
            
        # Bloc 3
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        # Classification
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 10. Compilation du modèle
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.summary()

    # 11. Entraînement
    print("\n--- Entraînement du modèle ---")
    print("(Cela peut prendre du temps sur CPU...)")
    epochs = 6
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 12. Sauvegarde du modèle
    print("\n--- Sauvegarde et Export (ZIP) ---")
    model_filename = "leaf_model.keras"
    model.save(model_filename)
    print(f"Modèle sauvegardé : {model_filename}")

    # 13. Création de l'archive ZIP
    # Le zip contient : le modèle + les images augmentées/modifiées
    zip_name = "dataset_and_model"
    print(f"Création de l'archive {zip_name}.zip en cours...")
    
    temp_dir = "temp_delivery"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Copier le modèle
    shutil.copy(model_filename, os.path.join(temp_dir, model_filename))
    
    # Copier le dataset augmenté (toutes les images transformées + augmentées)
    dataset_folder_name = "augmented_dataset"
    destination_dataset = os.path.join(temp_dir, dataset_folder_name)
    os.makedirs(destination_dataset, exist_ok=True)
    
    # Copier train et val
    shutil.copytree(str(train_final), os.path.join(destination_dataset, "train"))
    shutil.copytree(str(val_final), os.path.join(destination_dataset, "val"))
    
    # Copier aussi les fichiers de features
    shutil.copy(str(train_features_csv), os.path.join(temp_dir, "train_features.csv"))
    shutil.copy(str(val_features_csv), os.path.join(temp_dir, "val_features.csv"))

    # Créer le ZIP
    shutil.make_archive(zip_name, 'zip', temp_dir)
    
    # Nettoyage des dossiers temporaires
    shutil.rmtree(temp_dir)
    shutil.rmtree(output_base)
    
    print(f"✅ SUCCÈS : L'archive '{zip_name}.zip' a été créée.")
    print(f"   Contient : {model_filename} + {dataset_folder_name}/ + features.csv")

    # 14. Génération du Hash SHA1
    print("\n--- Génération de signature.txt ---")
    zip_filename = f"{zip_name}.zip"
    
    sha1 = hashlib.sha1()
    with open(zip_filename, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha1.update(data)
            
    hash_result = sha1.hexdigest()
    
    signature_file = "signature.txt"
    with open(signature_file, 'w') as f:
        f.write(hash_result)
        
    print(f"Hash SHA1 calculé : {hash_result}")
    print(f"✅ SUCCÈS : '{signature_file}' a été créé.")
    print("\n🎉 Entraînement terminé avec succès !")

if __name__ == "__main__":
    main()

