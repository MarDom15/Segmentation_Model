#!/usr/bin/env python
# coding: utf-8

# In[1]:
# import libraries such as OpenCV, NumPy, Matplotlib, Segmentation Models (an image segmentation library), TensorFlow, Keras, etc.

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
import random
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler  # Importing tools necessary for data preparation, such as pixel 
                                                # resizing and converting masks into categories.
scaler = MinMaxScaler()
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator # This line imports the ImageDataGenerator class from TensorFlow to augment data during training.
from keras.models import save_model
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style



# In[3]:
# definition of paths for directories containing training images and associated masks

train_img_dir = "/home/student/Dokumente/Unet_rest/data_for_keras_aug/train_images/train/"
train_mask_dir = "/home/student/Dokumente/Unet_rest/data_for_keras_aug/train_masks/train/"


# In[4]:

# list the files present in the paths defined above 
img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)


# In[5]:

# evaluation of the number of images in the folder 
num_images = len(os.listdir(train_img_dir))


# In[8]:

# random selection of a training image, read with OpenCV, and displayed alongside its corresponding mask for 
#visual verification of the effectiveness of the match.
img_num = random.randint(0, num_images-1)

img_for_plot = cv2.imread(train_img_dir+img_list[img_num], 1)
img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

mask_for_plot =cv2.imread(train_mask_dir+msk_list[img_num], 0)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.show()


# In[9]:

# These lines define the seed for generating random numbers, the batch size 
#and the number of classes in the data.
seed=24
batch_size= 8
n_classes=5

# In[11]:


#Use this to preprocess input for transfer learning
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


# In[12]:


#Define a function to perform additional preprocessing after datagen.
#For example, scale images, convert masks to categorical, etc. 
def preprocess_data(img, mask, num_class):
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
    #Convert mask to one-hot
    mask = to_categorical(mask, num_class)
      
    return (img,mask)


# In[14]:

# This function defines a generator for training data using ImageDataGenerator with various image augmentations.
def trainGenerator(train_img_path, train_mask_path, num_class):
    
    img_data_gen_args = dict(horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')
    
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)


# In[15]:

# These lines define the paths for the training images and masks directories and create a training generator.
train_img_path = "/home/student/Dokumente/Unet_rest/data_for_keras_aug/train_images/"
train_mask_path = "/home/student/Dokumente/Unet_rest/data_for_keras_aug/train_masks/"
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=5)


# In[16]:

# These lines define the paths for the training images and masks directories and create a validation generator.
val_img_path = "/home/student/Dokumente/Unet_rest/data_for_keras_aug/val_images/"
val_mask_path = "/home/student/Dokumente/Unet_rest/data_for_keras_aug/val_masks/"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=5)


# In[17]:


# Extraction of the first batch of images and masks from the training generator.
x, y = train_img_gen.__next__()


# In[18]:

# Displaying some examples of images and masks for verification.
for i in range(0,3):
    image = x[i]
    mask = np.argmax(y[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()


# In[19]:

# Extraction of the first batch of images and masks from the validation generator, then displaying them.
x_val, y_val = val_img_gen.__next__()

for i in range(0,3):
    image = x_val[i]
    mask = np.argmax(y_val[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()


# In[21]:


#Define the model metrcis and load model. 

num_train_imgs = len(os.listdir('/home/student/Dokumente/Unet_rest/data_for_keras_aug/train_images/train/'))
num_val_images = len(os.listdir('/home/student/Dokumente/Unet_rest/data_for_keras_aug/val_images/val/'))
steps_per_epoch = num_train_imgs//batch_size
val_steps_per_epoch = num_val_images//batch_size


# In[22]:

# Defining the dimensions of the input images for the model and class.
IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]
n_classes=5


# In[24]:


# Defining the loss function.
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    return 1 - (numerator + 1) / (denominator + 1)


# In[25]:


# Defining the focal loss function.
def focal_loss(y_true, y_pred, gamma=2, alpha=1):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_loss = -alpha * K.pow(1 - pt, gamma) * K.log(pt)
    return K.mean(focal_loss)


# In[26]:


# Defines the total loss as the sum of the Dice loss and the focal loss.
def total_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)


# In[27]:


# Intersection over Union (IoU) coefficient function.
def jacard_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth))


# In[ ]:

# This function defines a loss that is the sum of focal loss and Dice loss.
def focal_plus_dice_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss()(y_true, y_pred)
    return dice + focal


# In[ ]:



###def mask_loss(y_true, y_pred):
    #mse = MeanSquaredError()
    #loss = mse(y_true, y_pred)
    #return loss####


# In[28]:

#Use transfer learning using pretrained encoder in the U-Net
# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', 
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                classes=n_classes, activation='softmax')
# Define the metrics
metrics = ['accuracy', jacard_coef, sm.metrics.iou_score]
#[sm.metrics.iou_score]
model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=metrics)


# In[29]:


#Other losses to try: categorical_focal_dice_loss, cce_jaccard_loss, cce_dice_loss, categorical_focal_loss

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
print(model.summary())
print(model.input_shape)


# In[30]:


# Fit the model
#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=len(X_train) // 16, validation_steps=len(X_train) // 16, epochs=100)
# Train the model. 
history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch)

# In[31]:

# After training, calculate these metrics on the validation data.
validation_metrics = model.evaluate(val_img_gen, steps=val_steps_per_epoch)


# In[32]:


# Displaying or saving the metric values
print("Validation Loss:", validation_metrics[0])
print("Validation Accuracy:", validation_metrics[1])
print("Validation IoU (Jaccard Coef):", validation_metrics[2])


# In[33]:



# Definition of the full path to the model.
modele_dossier = "/home/student/Dokumente/Unet_rest/"
nom_du_modele = "landcover_300_epochs_RESNET_backbone_batch16.hdf5"
chemin_complet = modele_dossier + nom_du_modele

#  save the model.
model.save(chemin_complet)

# Displaying the full path.
print("The model has been saved in the folder:", modele_dossier)
print("Model name:", nom_du_modele)


# In[36]:


"""from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score


def total_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

# Charger le modèle avec les métriques personnalisées
model = load_model("C:/Users/mdomc/OneDrive/Dokumente/Unet_seg1/landcover_1_epochs_RESNET_backbone_batch16.hdf5",
                   custom_objects={
                       'total_loss': total_loss,
                       'jacard_coef': jacard_coef
                   })"""

#model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=metrics)


# In[2]:

# Loading the model.
model = load_model(".hdf5", compile=False)
model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=metrics)


# In[ ]:


""""from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Chemins vers les données de test
test_img_path = "/home/student/Dokumente/Unet_seg1/test/images/"
test_mask_path = "/home/student/Dokumente/Unet_seg1/test/masks/"

# Créer un générateur de données de test
test_datagen = ImageDataGenerator(rescale=1.0/255)  # Réduire l'échelle des pixels (0-255) à (0-1)

# Chargez les images de test à partir du répertoire
test_generator = test_datagen.flow_from_directory(
    test_img_path,
    target_size=(224, 224),  # Définissez la taille d'entrée de votre modèle
    class_mode=None,  # Aucune classification, car c'est un problème de segmentation
    shuffle=False  # Ne pas mélanger les images pour l'évaluation
)

# Chargez les masques de test à partir du répertoire
test_mask_generator = test_datagen.flow_from_directory(
    test_mask_path,
    target_size=(224, 224),  # Assurez-vous qu'il correspond à la taille d'entrée
    class_mode=None,
    color_mode='grayscale',  # Chargez les images en niveaux de gris
    shuffle=False
)

# Évaluez le modèle sur les données de test
test_metrics = model.evaluate(test_generator, steps=100)

# Affichez ou enregistrez les mesures d'évaluation
print("Test Loss:", test_metrics[0])
print("Test IoU (Jaccard Coef):", test_metrics[1])"""


# In[42]:


"""# Définir un générateur pour les données de test ou de validation
test_img_path = "/home/student/Dokumente/U-net/data_for_training_and_testing/val/images/"
test_mask_path = "/home/student/Dokumente/U-net/data_for_training_and_testing/val/masks/"
test_img_gen = trainGenerator(test_img_path, test_mask_path, num_class=5)
test_loss = model.evaluate(test_images, test_mask)

# Évaluer le modèle sur les données de test
test_metrics = model.evaluate(test_img_gen, steps=100)

# Afficher ou enregistrer les mesures d'évaluation
print("Test Mask Loss:", test_loss)
print("Test Loss:", test_metrics[0])
print("Test Accuracy:", test_metrics[1])
print("Test IoU (Jaccard Coef):", test_metrics[2])"""


# In[43]:


# Évaluation du modèle
#y_pred = model.predict(X_test)
#y_pred_argmax = np.argmax(y_pred, axis=-1)
#y_test_argmax = np.argmax(y_test, axis=-1)


# Test the model and obtain predictions.
test_image_batch, test_mask_batch = val_img_gen.__next__()

# Convert categories to integers for visualization and IoU calculation.
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3)
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)


# Obtenez les variables y_test_argmax et y_pred_argmax
y_test_argmax = test_mask_batch_argmax
y_pred_argmax = test_pred_batch_argmax

# In[47]:

# Defining the class names corresponding to the indices.
class_names = ["unlabeled", "tree", "building", "water", "freespace"] #backgrund
class_colors = {
    0: (255, 0, 0),    # Rouge (Building)
    1: (173, 216, 230),  # Vert clair (Freespace)
    2: (0, 128, 0),    # Vert foncé (Tree)
    3: (0, 0, 255),    # Bleu (Water)
    4: (169, 169, 169)  # Gris (Unlabeled)
}

# Calculation of metrics per class.
precision_per_class = precision_score(y_test_argmax.ravel(), y_pred_argmax.ravel(), average=None)
recall_per_class = recall_score(y_test_argmax.ravel(), y_pred_argmax.ravel(), average=None)
f1_per_class = f1_score(y_test_argmax.ravel(), y_pred_argmax.ravel(), average=None)
confusion = confusion_matrix(y_test_argmax.ravel(), y_pred_argmax.ravel())


# Preparing the data for the table.
data = []
for class_idx, class_name in enumerate(class_names):
    precision_class = precision_per_class[class_idx]
    recall_class = recall_per_class[class_idx]
    f1_class = f1_per_class[class_idx]
    color = class_colors.get(class_name, np.array([0, 0, 0]))  # Default color is black
    data.append([
        "\033[1m" + class_name + "\033[0m",
        precision_class,
        recall_class,
        f1_class,
        "\033[38;2;{};{};{}m".format(*color) + "███" + "\033[0m"  # Display color as a color block
    ])

data.append([
    "\033[1m" + "Confusion Matrix:" + "\033[0m",
    "",
    "",
    "",
    "\n" + str(confusion)
])

# Displaying the formatted table
table_headers = ["Classe", "Precision", "Recall", "F1-Score", "Couleur", "IuO"]
table = tabulate(data, headers=table_headers, tablefmt="grid")
print(table)


# In[48]:

# Calculating metrics for the overall/general case.
precision = precision_score(y_test_argmax.ravel(), y_pred_argmax.ravel(), average='macro')
recall = recall_score(y_test_argmax.ravel(), y_pred_argmax.ravel(), average='macro')
f1 = f1_score(y_test_argmax.ravel(), y_pred_argmax.ravel(), average='macro')
confusion = confusion_matrix(y_test_argmax.ravel(), y_pred_argmax.ravel())

# Preparing the data for the table.
data = [
    ["\033[1m" + "Precision:" + "\033[0m", precision],
    ["\033[1m" + "Recall:" + "\033[0m", recall],
    ["\033[1m" + "F1-Score:" + "\033[0m", f1],
    ["\033[1m" + "Confusion Matrix:" + "\033[0m", "\n" + str(confusion)]
]

# Afficher le tableau formaté
table = tabulate(data, headers=["Métrique", "Valeur"], tablefmt="grid")
print(table)



# In[50]:


# Calculating the confusion matrix (replace y_true and y_pred with your actual and predicted values).
y_true = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
y_pred = np.array([0, 1, 1, 3, 4, 2, 0, 0, 2, 3, 4, 5])

# Defining the class names corresponding to the indices.
class_names = ["building", "freespace", "tree", "water", "unlabeled"] #backgrund

# Creating a confusion matrix.
confusion = confusion_matrix(y_true, y_pred)

# Calculer les pourcentages normalisés
confusion_percent = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

# Créer une colormap personnalisée avec des couleurs pour chaque classe
#colors = ['red', 'green', 'yellow', 'blue', 'brown']
#cmap = LinearSegmentedColormap.from_list("Custom", colors, N=len(colors))

# Creating a custom chart.
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_percent, annot=True, fmt=".2%", cmap=cmap, cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matrice de Confusion Normalisée")
plt.xlabel("Prédictions")
plt.ylabel("Vraies étiquettes")
plt.show()


# In[52]:


acc = history.history['iou_score']
val_acc = history.history['val_iou_score']


# In[53]:


#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[54]:


# Tracer la courbe de perte du masque (loss_mask)
"""loss_mask = history.history['loss_mask']
val_loss_mask = history.history['val_loss_mask']

plt.plot(epochs, loss_mask, 'b', label='Training loss_mask')
plt.plot(epochs, val_loss_mask, 'g', label='Validation loss_mask')
plt.title('Training and validation loss_mask')
plt.xlabel('Epochs')
plt.ylabel('Loss Mask')
plt.legend()
plt.show()"""


# In[55]:


# Plotting the accuracy curve.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[57]:


# Plotting the IoU (Intersection over Union) curve.
iou = history.history['jacard_coef']  # Utilisez 'jacard_coef' ici
val_iou = history.history['val_jacard_coef']  # Utilisez 'val_jacard_coef' ici

plt.plot(epochs, iou, 'y', label='Training IoU')
plt.plot(epochs, val_iou, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()


# In[58]:


plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()


# In[59]:


# Supposons que y_test_argmax et y_pred_argmax soient vos données de test et de prédiction respectivement
conf_matrix = confusion_matrix(y_test_argmax.flatten(), y_pred_argmax.flatten())
n_classes = conf_matrix.shape[0]

# Mapping dictionary between class indices and label names.
class_labels = {
    0: 'Building',
    1: 'freespace',
    2: 'tree',
    3: 'water',
    4: 'unlabeled'   #backgrund
}

iou_per_class = []
for i in range(n_classes):
    intersection = conf_matrix[i, i]
    union = np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - intersection
    iou = intersection / union if union != 0 else 0
    iou_per_class.append(iou)

table = []

for i in range(n_classes):
    class_name = class_labels[i]
    table.append([class_name, f"{iou_per_class[i]:.4f}"])

# Affichage du tableau avec des valeurs en gras
table_headers = [Fore.GREEN + "Class" + Style.RESET_ALL, Fore.GREEN + "IoU" + Style.RESET_ALL]
print(tabulate(table, headers=table_headers, tablefmt="grid"))


# In[60]:


#batch_size=32 #Check IoU for a batch of images

#Test generator using validation data.

test_image_batch, test_mask_batch = val_img_gen.__next__()


# In[61]:


#Convert categorical to integer for visualization and IoU calculation
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)


# In[62]:


n_classes = 5
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


# In[63]:



#View a few images, masks and corresponding predictions. 
img_num = random.randint(0, test_image_batch.shape[0]-1)


# In[64]:


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image_batch[img_num])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_batch_argmax[img_num])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_pred_batch_argmax[img_num])
plt.show()


# In[ ]:




