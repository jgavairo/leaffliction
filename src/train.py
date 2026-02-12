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
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from Transformation import Transformation
from augment import augment_class

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32


def extract_base_name(filename):
    """
    Extrait le base name d'un fichier en supprimant les suffixes
    d'augmentation.
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
    Groupe les images par leur base name
    (image originale + ses augmentations).

    Returns: dict {base_name: [list of image paths]}
    """
    img_extensions = {".jpg", ".jpeg", ".png"}
    images = [
        f for f in class_dir.iterdir()
        if f.suffix.lower() in img_extensions
    ]

    groups = defaultdict(list)
    for img_path in images:
        base_name = extract_base_name(img_path.name)
        groups[base_name].append(img_path)

    return groups


def split_by_groups(class_dir, train_ratio=0.8, seed=123):
    """
    Split les images en train/val en respectant les groupes
    (image mère + augmentations).

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


def process_single_image(args):
    """
    Fonction isolée pour traiter une seule image en parallèle.

    Args:
        args: tuple (img_path, masks_out, norm_out, orig_out, img_width,
                     img_height, save_original)

    Returns:
        dict: features de l'image ou None en cas d'erreur
    """
    (
        img_path, masks_out, norm_out, orig_out,
        img_width, img_height, save_original
    ) = args

    try:
        # Lire l'image
        image = cv.imread(str(img_path))
        if image is None:
            return None

        # Appliquer la transformation
        transformer = Transformation(image)
        masked = transformer.masked_leaf()
        mask = transformer.mask

        # Calculer les features
        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
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
        mean_r = float(mean_bgr[2])
        mean_g = float(mean_bgr[1])
        mean_b = float(mean_bgr[0])

        # Préparer les noms de fichiers
        base_name = Path(img_path).stem
        ext = Path(img_path).suffix
        mask_name = f"{base_name}_mask.png"
        norm_name = f"{base_name}_norm{ext}"

        # Sauvegarder le masque
        mask_path = Path(masks_out) / mask_name
        cv.imwrite(str(mask_path), mask)

        # Sauvegarder l'image normalisée redimensionnée
        norm = cv.resize(
            masked, (img_width, img_height),
            interpolation=cv.INTER_AREA
        )
        norm_path = Path(norm_out) / norm_name
        cv.imwrite(str(norm_path), norm)

        # Sauvegarder l'image originale redimensionnée (val)
        if save_original and orig_out:
            orig_name = f"{base_name}{ext}"
            orig_resized = cv.resize(
                image, (img_width, img_height),
                interpolation=cv.INTER_AREA
            )
            orig_path = Path(orig_out) / orig_name
            cv.imwrite(str(orig_path), orig_resized)

        # Retourner les features
        return {
            "file": str(img_path),
            "mask_path": str(mask_path),
            "norm_path": str(norm_path),
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

    except Exception as e:
        print(f"  Erreur avec {Path(img_path).name}: {e}")
        return None


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
        split_data[class_dir.name] = split_by_groups(
            class_dir, train_ratio=0.8, seed=123
        )
        train_count = len(split_data[class_dir.name]['train'])
        val_count = len(split_data[class_dir.name]['val'])
        print(f"{class_dir.name}: {train_count} train, {val_count} val")

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

    # 4. Transformation des images TRAIN (parallélisée)
    print("\n--- Transformation des images TRAIN (parallèle) ---")
    train_features_csv = output_train / "features.csv"
    csv_fieldnames = [
        "file", "mask_path", "norm_path", "area", "perimeter",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "mean_r", "mean_g", "mean_b"
    ]

    # Créer le CSV avec header
    with open(train_features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()

    train_transformed = output_train / "transformed"
    train_transformed.mkdir(parents=True, exist_ok=True)

    # Préparer les tâches pour le pool
    train_tasks = []
    for class_name, splits in split_data.items():
        # Créer les dossiers de sortie pour cette classe
        masks_out = train_transformed / "masks" / class_name
        norm_out = train_transformed / "normalized" / class_name
        masks_out.mkdir(parents=True, exist_ok=True)
        norm_out.mkdir(parents=True, exist_ok=True)

        for img_path in splits['train']:
            task = (
                str(img_path), str(masks_out), str(norm_out), None,
                IMG_WIDTH, IMG_HEIGHT, False
            )
            train_tasks.append(task)

    # Exécution parallèle avec barre de progression
    print(f"Traitement de {len(train_tasks)} images TRAIN...")
    all_train_features = []

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_single_image, train_tasks),
            total=len(train_tasks),
            desc="Train transformation"
        ))

    # Filtrer les résultats valides et écrire dans le CSV
    all_train_features = [r for r in results if r is not None]

    with open(train_features_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writerows(all_train_features)

    total_train = len(all_train_features)
    print(f"Total d'images TRAIN transformées : {total_train}")

    # 5. Transformation des images VALIDATION (parallélisée)
    print("\n--- Transformation des images VALIDATION (parallèle) ---")
    val_features_csv = output_val / "features.csv"

    # Créer le CSV avec header
    with open(val_features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()

    val_transformed = output_val / "transformed"
    val_transformed.mkdir(parents=True, exist_ok=True)

    # Aussi créer un dossier pour les images originales redimensionnées
    val_original = output_val / "original_resized"
    val_original.mkdir(parents=True, exist_ok=True)

    # Préparer les tâches pour le pool
    val_tasks = []
    for class_name, splits in split_data.items():
        # Créer les dossiers de sortie pour cette classe
        masks_out = val_transformed / "masks" / class_name
        norm_out = val_transformed / "normalized" / class_name
        orig_out = val_original / class_name
        masks_out.mkdir(parents=True, exist_ok=True)
        norm_out.mkdir(parents=True, exist_ok=True)
        orig_out.mkdir(parents=True, exist_ok=True)

        for img_path in splits['val']:
            task = (
                str(img_path), str(masks_out), str(norm_out), str(orig_out),
                IMG_WIDTH, IMG_HEIGHT, True
            )
            val_tasks.append(task)

    # Exécution parallèle avec barre de progression
    print(f"Traitement de {len(val_tasks)} images VALIDATION...")
    all_val_features = []

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_single_image, val_tasks),
            total=len(val_tasks),
            desc="Val transformation"
        ))

    # Filtrer les résultats valides et écrire dans le CSV
    all_val_features = [r for r in results if r is not None]

    with open(val_features_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writerows(all_val_features)

    total_val = len(all_val_features)
    print(f"Total d'images VALIDATION transformées : {total_val}")

    # 6. Augmentation MODÉRÉE des images TRAIN (seulement petites classes)
    print("\n--- Augmentation modérée des petites classes ---")
    train_normalized = train_transformed / "normalized"
    train_augmented = output_train / "augmented"
    train_augmented.mkdir(parents=True, exist_ok=True)

    # Compter les images par classe dans TRAIN
    class_counts = {}
    for class_dir in train_normalized.iterdir():
        if class_dir.is_dir():
            img_count = len([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in img_extensions
            ])
            class_counts[class_dir] = img_count

    # Augmentation MODÉRÉE: seulement classes avec <600 images
    # On augmente jusqu'à min(800, max_count * 0.6) pour éviter suraugmentation
    if class_counts:
        max_count = max(class_counts.values())
        # Target: 60% du max ou 800, selon le plus petit
        moderate_target = min(800, int(max_count * 0.6))
        print(
            f"Classe max: {max_count} images, "
            f"target modéré: {moderate_target}"
        )

        for class_dir, count in class_counts.items():
            # Seulement les classes avec <600 images originales
            if count < 600:
                # Augmenter jusqu'au target modéré (mais pas au-delà)
                target = min(moderate_target, count * 3)
                deficit = max(0, target - count)
                if deficit > 0:
                    msg = (
                        f"Augmentation modérée de {class_dir.name}: "
                        f"{count} -> {count + deficit} (+{deficit})"
                    )
                    print(msg)
                    augment_class(
                        class_dir, deficit, train_augmented, verbose=True
                    )
                else:
                    print(f"Pas d'augmentation pour {class_dir.name}")
            else:
                print(f"Classe suffisante: {class_dir.name} ({count})")

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

    # 8. Préparer le dataset VALIDATION
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

    # 9. Validation des données transformées
    print("\n--- Validation des données transformées ---")
    for split_name, split_dir in [("Train", train_final), ("Val", val_final)]:
        print(f"Vérification de {split_name}:")
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            images = list(class_dir.iterdir())
            if not images:
                print(f"  ⚠ {class_dir.name}: AUCUNE IMAGE!")
                continue

            # Vérifier quelques images aléatoires
            sample_imgs = random.sample(images, min(3, len(images)))
            corrupted = 0
            for img_path in sample_imgs:
                try:
                    img = cv.imread(str(img_path))
                    if img is None or img.size == 0:
                        corrupted += 1
                except Exception:
                    corrupted += 1

            if corrupted > 0:
                msg = f"  ⚠ {class_dir.name}: {corrupted}/"
                msg += f"{len(sample_imgs)} images corrompues!"
                print(msg)
            else:
                print(f"  ✓ {class_dir.name}: {len(images)} images OK")

    # 10. Chargement des datasets (pas de split, déjà séparés)
    print("\n--- Préparation des datasets ---")
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

    # 10. Vérification et calcul des class weights
    class_names = train_ds.class_names
    print(f"\nClasses trouvées : {class_names}")
    print(f"Nombre de classes : {len(class_names)}")

    # Calculer les poids de classe pour gérer le déséquilibre
    print("\n--- Calcul des class weights ---")
    class_image_counts = {}
    for class_dir in train_final.iterdir():
        if class_dir.is_dir():
            count = len([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in img_extensions
            ])
            class_image_counts[class_dir.name] = count
            print(f"  {class_dir.name}: {count} images")

    # Calculer les poids (inverse de la fréquence)
    total_images = sum(class_image_counts.values())
    class_weight = {}
    for idx, class_name in enumerate(class_names):
        count = class_image_counts.get(class_name, 1)
        weight = total_images / (len(class_names) * count)
        class_weight[idx] = weight
        print(f"  Classe {idx} ({class_name}): poids = {weight:.3f}")

    # 11. Optimisation des performances
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(
        buffer_size=AUTOTUNE
    )
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    # 12. Construction du modèle avec Transfer Learning (MobileNetV2)
    msg = "\n--- Construction du modèle avec Transfer Learning "
    msg += "(MobileNetV2) ---"
    print(msg)

    # Charger MobileNetV2 pré-entraîné sur ImageNet sans le top
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )

    # Geler les poids du modèle de base
    base_model.trainable = False
    msg = "Poids du modèle de base gelés: "
    msg += f"{len(base_model.layers)} layers"
    print(msg)

    # Construire le modèle complet
    # Note: MobileNetV2 preprocess_input fait: (x / 127.5) - 1.0
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 13. Compilation du modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        ),
        metrics=['accuracy']
    )

    model.summary()

    # 14. Entraînement
    print("\n--- Entraînement du modèle ---")
    print("(Cela peut prendre du temps sur CPU...)")

    # Callbacks pour améliorer l'entraînement
    callbacks = [
        # Early stopping: arrête si val_accuracy ne s'améliore pas
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate si plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    epochs = 30
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight  # Gère le déséquilibre
    )

    # 15. Sauvegarde du modèle
    print("\n--- Sauvegarde et Export (ZIP) ---")
    model_folder = "dataset_and_model"
    os.makedirs(model_folder, exist_ok=True)
    model_filename = os.path.join(model_folder, "leaf_model.keras")
    model.save(model_filename)
    print(f"Modèle sauvegardé : {model_filename}")

    # 16. Création de l'archive ZIP
    # Le zip contient : le modèle + les images augmentées/modifiées
    zip_name = "dataset_and_model"
    print(f"Création de l'archive {zip_name}.zip en cours...")

    temp_dir = "temp_delivery"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Copier le modèle
    shutil.copy(
        model_filename, os.path.join(temp_dir, "leaf_model.keras")
    )

    # Copier le dataset augmenté
    dataset_folder_name = "augmented_dataset"
    destination_dataset = os.path.join(temp_dir, dataset_folder_name)
    os.makedirs(destination_dataset, exist_ok=True)

    # Copier train et val
    shutil.copytree(
        str(train_final), os.path.join(destination_dataset, "train")
    )
    shutil.copytree(
        str(val_final), os.path.join(destination_dataset, "val")
    )

    # Copier aussi le dataset de validation original (non normalisé)
    shutil.copytree(
        str(val_original),
        os.path.join(destination_dataset, "val_original")
    )

    # Copier aussi les fichiers de features
    shutil.copy(
        str(train_features_csv),
        os.path.join(temp_dir, "train_features.csv")
    )
    shutil.copy(
        str(val_features_csv),
        os.path.join(temp_dir, "val_features.csv")
    )

    # Créer le ZIP
    shutil.make_archive(zip_name, 'zip', temp_dir)

    # Nettoyage des dossiers temporaires
    shutil.rmtree(temp_dir)
    shutil.rmtree(output_base)

    print(f"✅ SUCCÈS : L'archive '{zip_name}.zip' a été créée.")
    msg = (
        f"   Contient : leaf_model.keras + {dataset_folder_name}/ + "
        "features.csv"
    )
    print(msg)

    # 17. Génération du Hash SHA1
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
