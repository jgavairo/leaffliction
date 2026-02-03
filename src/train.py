import sys
import os
import tensorflow as tf
from tensorflow.keras import layers

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32

def main():
    print("Script démarré...")
    print(f"Arguments: {sys.argv}")

    # 1 Verification des args
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_dataset>")
        return

    data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print(f"Erreur : Le dossier '{data_dir}' n'existe pas.")
        return
    
    print(f"chargement des donnees depuis : {data_dir}")

    # 2 creation du jeu d'entrainement (80% des images)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # 3 Création du jeu de validation (20% des images)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",     # Ici on prend la partie Validation
        seed=123,                # TRES IMPORTANT: le même seed que pour le training !
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # 4 verification
    class_names = train_ds.class_names
    print(f"\nClasses trouvees : {class_names}")
    print(f"Nombre de classes : {len(class_names)}")

    # 5 Optimisation des performances pour eviter de charger les donees a chaque epoch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    #6 construction du modele avec CNN
    print("\nConstruction du modele...")
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
        tf.keras.layers.Dense(num_classes, activation='softmax') # Softmax pour la classification multi-classes
    ])

    # A REVOIR !!!

    # 7. Compilation du modèle
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.summary() # Affiche un résumé de l'architecture

    # 8. Entraînement
    print("\nDébut de l'entraînement (Cela peut prendre du temps sur CPU)...")
    epochs = 10 # Tu pourras augmenter ce chiffre si la précision n'est pas suffisante
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 9. Sauvegarde du modèle (Format Keras)
    print("\nSauvegarde du modèle...")
    model_filename = "leaf_model.keras"
    model.save(model_filename)
    print(f"Modèle sauvegardé sous : {model_filename}")

if __name__ == "__main__":
    main()

