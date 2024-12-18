#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Programm zum Teilen von Bildern und Masken in einer bestimmten 

#import cv2,  Importiert das OpenCV-Modul für die Bildverarbeitung in Python.
# Importiert das os-Modul für die Interaktion mit dem Betriebssystem und Dateiverwaltung in Python.

import cv2
import os

# Definition des Pfads zum Verzeichnis der Bilder und Masken.
source_image_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/images"
source_mask_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/masks"

# Der Pfad zum Verzeichnis, in dem Sie die zugeschnittenen Bilder und Masken speichern möchten.
output_image_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/images_256"
output_mask_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/masks_256"

# Erstellen Sie das Ausgabeverzeichnis, falls es nicht existiert
os.makedirs(output_image_directory, exist_ok=True)

# Definition der gewünschten Größe der Bilder (256256, 512512, ...).
piece_size = 256

# Überprüfung der Anwesenheit von Fotos in den Ordnern
for index, image_filename in enumerate(os.listdir(source_image_directory)):
    if image_filename.endswith(".tiff"):  # Überprüfung der verschiedenen Dateierweiterungen."
        image_path = os.path.join(source_image_directory, image_filename)
        mask_filename = image_filename.replace(".tiff", ".png")
        mask_path = os.path.join(source_mask_directory, mask_filename)

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        height, width, _ = img.shape

        # Teilung der Bilder und Masken in die zuvor definierte Größe.
        for y in range(0, height, piece_size):
            for x in range(0, width, piece_size):
                piece_img = img[y:y + piece_size, x:x + piece_size]
                piece_mask = mask[y:y + piece_size, x:x + piece_size]

                # Überprüfung der Bildgröße, um solche zu löschen, die nicht die angemessene Größe haben.
                if piece_img.shape[0] == piece_size and piece_img.shape[1] == piece_size:
                    # Umbenennung der neuen Bilder."

                    piece_filename = f"image_{index:04d}_{y:04d}_{x:04d}.tiff"
                    piece_path = os.path.join(output_image_directory, piece_filename)

                    mask_piece_filename = f"mask_{index:04d}_{y:04d}_{x:04d}.png"
                    mask_piece_path = os.path.join(output_mask_directory, mask_piece_filename)

                    cv2.imwrite(piece_path, piece_img)
                    cv2.imwrite(mask_piece_path, piece_mask)



# In[26]:

#Ein weiteres Programm zum Umbenennen".
import os

image_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/images_256"
mask_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/masks_256"

def rename_files(directory, base_name):
    file_list = sorted(os.listdir(directory))
    
    for i, filename in enumerate(file_list):
        _, file_extension = os.path.splitext(filename)
        new_filename = f"{base_name}_{i:04d}{file_extension}"
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

# Umbenennen der Bilder im Verzeichnis 
rename_files(image_directory, "image")

# Umbenennen der Bilder im Verzeichnis 'masks'
rename_files(mask_directory, "image")


# In[27]:


"""import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np

# Chemin vers le répertoire contenant vos images découpées
image_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/images_256"
mask_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/masks_256"

# Obtenez la liste des fichiers d'images dans le répertoire
image_files = [file for file in os.listdir(image_directory) if file.endswith(".tiff")]

# Sélection aléatoire de 20 images
random_images = random.sample(image_files, 20)

# Créez une nouvelle figure
plt.figure(figsize=(12, 10))

for i, image_file in enumerate(random_images):
    image_path = os.path.join(image_directory, image_file)

    # Charger l'image
    image = cv2.imread(image_path)

    if image is not None:
        # Charger le masque
        mask_filename = image_file.replace(".tiff", ".png")
        mask_path = os.path.join(mask_directory, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            # Convertir les données en tableaux NumPy avec des valeurs de pixels en virgule flottante
            image = image.astype(np.float32) / 255.0  # Conversion à une échelle de 0 à 1
            mask = mask.astype(np.float32) / 255.0  # Conversion à une échelle de 0 à 1

            # Afficher l'image
            plt.subplot(10, 4, i * 2 + 1)
            plt.imshow(image, cmap="gray")
            plt.title("Image")

            # Afficher le masque
            plt.subplot(10, 4, i * 2 + 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Masque")

plt.tight_layout()
plt.show()"""


# In[28]:
# Dieses Programm erstellt Ordner und teilt die Daten in Trainings- und Validierungsdaten auf.

# Importiert das Modul random, das Funktionen im Zusammenhang mit der Erzeugung von Zufallszahlen bereitstellt.
# Importiert das Modul shutil, das Operationen auf Dateien und Verzeichnissen auf höherer Ebene bereitstellt, einschließlich Kopieren, Verschieben und Löschen

import os
import random
import shutil

# Der Pfad zum Verzeichnis, das Ihre Bilder und Masken enthält.
image_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/images_256"
mask_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/masks_256"

# Erstellung der Trainings- und Validierungsverzeichnisse, wenn sie nicht vorhanden sind
train_images_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/train_images"
train_masks_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/train_masks"
val_images_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/val_images"
val_masks_directory = "C:/Users/mdomc/OneDrive/Bureau/Prog/val_masks"


# Erstellung der Verzeichnisse, falls sie nicht existieren.
os.makedirs(train_images_directory, exist_ok=True)
os.makedirs(train_masks_directory, exist_ok=True)
os.makedirs(val_images_directory, exist_ok=True)
os.makedirs(val_masks_directory, exist_ok=True)


# Teilungsverhältnis"
train_ratio = 0.80  # 70% pour l'entraînement
val_ratio = 0.20  # 15% pour la validation


# Erhalten der Liste der Bilddateien im Verzeichnis.
image_files = [file for file in os.listdir(image_directory) if file.endswith(".tiff")]

# Mélangez la liste pour une répartition aléatoire
random.shuffle(image_files)

# Teilung der Bilder in Trainings- und Validierungs
total_images = len(image_files)
train_split = int(train_ratio * total_images)
val_split = int(val_ratio * total_images)

train_images = image_files[:train_split]
val_images = image_files[train_split:(train_split + val_split)]


# Verschieben der Bilder und Masken in die entsprechenden Verzeichnisse.
def move_images_and_masks(image_files, source_image_dir, source_mask_dir, dest_image_dir, dest_mask_dir):
    for image_file in image_files:
        image_path = os.path.join(source_image_dir, image_file)
        mask_path = os.path.join(source_mask_dir, image_file.replace(".tiff", ".png"))
        shutil.move(image_path, os.path.join(dest_image_dir, image_file))
        shutil.move(mask_path, os.path.join(dest_mask_dir, image_file.replace(".tiff", ".png")))

move_images_and_masks(train_images, image_directory, mask_directory, train_images_directory, train_masks_directory)
move_images_and_masks(val_images, image_directory, mask_directory, val_images_directory, val_masks_directory)



# In[ ]:




