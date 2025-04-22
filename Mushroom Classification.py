#!/usr/bin/env python
# coding: utf-8

# # Mushroom Classification Machine Learning - Neural Models
#  
#  
#  Kiara Zepeda

# Dataset Overview:
# 
# - Kaggle Data Source (https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images/data)
# - Contains nine folders with 300-1500 images each, almost 2 GB of storage.
# - Images of most common Northern European mushroom genuses
# 
# The Objective
# - Accurately classify mushrooms into their respective genus. 

# In[1]:


import zipfile
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# To fix "Image File is truncated" error during training
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# There were too many images, I had to upload the data as a zip

# In[3]:


zip_file_path = 'mushroom.zip'


# In[4]:


extracted_folder = 'extracted_data'

# Check if folder exists, create if not
if not os.path.exists(extracted_folder):
    os.makedirs(extracted_folder)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

print(f"ZIP file extracted to {extracted_folder}")


# In[5]:


extracted_files = os.listdir(extracted_folder)
print(extracted_files)


# In[6]:


# Checking the structure of the folders
mushrooms_folder = os.path.join(extracted_folder, 'Mushrooms')
mushrooms_folder_contents = os.listdir(mushrooms_folder)
print(f"Contents of 'Mushrooms' folder: {mushrooms_folder_contents}")

mushrooms_lowercase_folder = os.path.join(extracted_folder, 'mushrooms')
mushrooms_lowercase_folder_contents = os.listdir(mushrooms_lowercase_folder)
print(f"Contents of 'mushrooms' folder: {mushrooms_lowercase_folder_contents}")


# In[7]:


import os

# Define the root directory where the mushroom folders are located
root_dir = 'extracted_data/Mushrooms'

# Loop through each subfolder in the root directory
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Count the number of image files in this folder
        image_count = 0
        for file in os.listdir(folder_path):
            # Check if the file is an image (based on file extension)
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_count += 1
        
        # Print the folder name and the count of images in it
        print(f'{folder}: {image_count} images')


# I want to reduce the size to only 5 genuses. So I chose the ones with a smallest amount of images. I also chose to reduce the size of the remaining 5 to only 500 images. That way, they are equal and it takes less processing time. 

# In[9]:


import random

root_dir = 'extracted_data/Mushrooms'
unwanted_genuses = ['Suillus', 'Hygrocybe', 'Agaricus', 'Entoloma']

# Define the path for the unified folder
unified_folder = 'unified_folder'

# Create the unified folder if it doesn't exist
os.makedirs(unified_folder, exist_ok=True)

remaining_genuses = []

# Loop through each subfolder (genus) in the root directory
for genus in os.listdir(root_dir):
    genus_path = os.path.join(root_dir, genus)

    # Skip the unwanted genuses
    if genus in unwanted_genuses or not os.path.isdir(genus_path):
        continue
    
    # Add to the remaining genus list
    remaining_genuses.append(genus)

# Ensure there are exactly 5 remaining genus
if len(remaining_genuses) != 5:
    print("ERROR: There should be exactly 5 remaining genera.")
else:
    print(f"Remaining genera: {remaining_genuses}")

# Process each remaining genus
for genus in remaining_genuses:
    genus_path = os.path.join(root_dir, genus)
    genus_images = []
    
    # Get a list of image files in the genus folder
    for file in os.listdir(genus_path):
        if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            genus_images.append(os.path.join(genus_path, file))
    
    # Randomly select 500 images (or all if there are fewer than 500)
    random.shuffle(genus_images)
    selected_images = genus_images[:500]

    # Create a folder for this genus in the unified folder
    genus_folder = os.path.join(unified_folder, genus)
    os.makedirs(genus_folder, exist_ok=True)
    
    # Copy the selected images to the genus folder in the unified folder
    for i, image_path in enumerate(selected_images):
        destination = os.path.join(genus_folder, f'image_{i+1}.jpg')
        shutil.copy(image_path, destination)

    print(f"Successfully copied 500 images for genus: {genus}")

print(f"Finished processing all genera. 500 random images have been copied to each genus folder.")


# Now, the images are in the correct folder and are sorted. Let's do some EDA. 

# In[14]:


# Path to the unified folder
unified_folder = 'unified_folder'

# List to store the number of images per genus
genus_image_counts = {}

# Loop through the genus folders and count the images
for genus in os.listdir(unified_folder):
    genus_path = os.path.join(unified_folder, genus)
    
    if os.path.isdir(genus_path):
        image_count = len([f for f in os.listdir(genus_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
        genus_image_counts[genus] = image_count

colors = ['red', 'green', 'blue', 'orange', 'purple'] 

plt.figure(figsize=(10, 6))
plt.bar(genus_image_counts.keys(), genus_image_counts.values(), color=colors[:len(genus_image_counts)]) 
plt.title('Number of Images per Genus')
plt.xlabel('Genus')
plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right')
plt.show()



# In[15]:


from PIL import Image

# Path to the unified folder
unified_folder = 'unified_folder'

# List to store one image path from each genus
sample_images = []

# Loop through the genus folders and select one random image from each
for genus in os.listdir(unified_folder):
    genus_path = os.path.join(unified_folder, genus)
    
    if os.path.isdir(genus_path):
        # Get a list of image files in the genus folder
        image_files = [f for f in os.listdir(genus_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        # Select one random image from the list
        if image_files:
            sample_image = random.choice(image_files)
            sample_images.append((genus, os.path.join(genus_path, sample_image)))

# Set up the grid for displaying the images
num_images = len(sample_images)
cols = 3  
rows = (num_images // cols) + (num_images % cols > 0)

# Create a figure with subplots
plt.figure(figsize=(15, 5 * rows))

# Loop through the sample images and display them
for i, (genus, image_path) in enumerate(sample_images):
    plt.subplot(rows, cols, i + 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')  
    plt.title(genus)

plt.tight_layout()
plt.show()


# In[17]:


# List to store image dimensions (width, height)
image_dimensions = []

# Loop through each genus folder and collect image dimensions
for genus in os.listdir(unified_folder):
    genus_path = os.path.join(unified_folder, genus)
    
    if os.path.isdir(genus_path):
        # Loop through each image in the genus folder
        for file in os.listdir(genus_path):
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_path = os.path.join(genus_path, file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_dimensions.append((width, height))

# Separate widths and heights
widths, heights = zip(*image_dimensions)
plt.figure(figsize=(14, 6))

# Histogram for widths
plt.subplot(1, 2, 1)
plt.hist(widths, bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Image Widths')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')

# Histogram for heights
plt.subplot(1, 2, 2)
plt.hist(heights, bins=30, color='green', edgecolor='black')
plt.title('Distribution of Image Heights')
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Scatter plot of Width vs. Height
plt.figure(figsize=(8, 6))
plt.scatter(widths, heights, alpha=0.5, color='purple')
plt.title('Image Dimensions (Width vs Height)')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.show()


# In[21]:


import numpy as np

image_sizes = []
aspect_ratios = []

for genus_name in os.listdir(unified_folder):
    genus_folder = os.path.join(unified_folder, genus_name)
    
    if os.path.isdir(genus_folder):
        for image_file in os.listdir(genus_folder):
            img_path = os.path.join(genus_folder, image_file)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    image_sizes.append((width, height))
                    aspect_ratios.append(aspect_ratio)
            except:
                continue  ]

# Convert image sizes to a numpy array for easy handling
image_sizes = np.array(image_sizes)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(image_sizes[:, 0], image_sizes[:, 1], alpha=0.5, c=aspect_ratios, cmap='viridis')
plt.title('Image Aspect Ratio Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.colorbar(scatter, label='Aspect Ratio')
plt.show()



# Non-Supervised Learning Model
# # K-Means Clustering
# 
# K-Means is a clustering algorithm that groups data points into K clusters based on their similarity. I want to extract features from the images and then apply K-Means clustering to see how images are grouped together.

# In[22]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
import random
from tensorflow.keras.models import Model


# In[23]:


# Feature Extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))  # Don't include the top layer
model = Model(inputs=base_model.input, outputs=base_model.output)

# Using VGG16, pre-trained Convolutional Neural Network (CNN) model


# In[33]:


#Image Resizing and Feature Extraction
def extract_features_from_folder(folder_path, target_size=(150, 150)):
    feature_list = []
    image_paths = []
    genus_labels = []

    for genus_name in os.listdir(folder_path):
        genus_folder = os.path.join(folder_path, genus_name)
        if os.path.isdir(genus_folder):
            for image_file in os.listdir(genus_folder):
                img_path = os.path.join(genus_folder, image_file)
                
                # Load image
                img = image.load_img(img_path, target_size=target_size)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Extract features
                features = model.predict(img_array)
                features = features.flatten()  # Flatten to 1D array

                feature_list.append(features)
                image_paths.append(img_path)
                genus_labels.append(genus_name)
    
    return np.array(feature_list), image_paths, genus_labels

# Extract features from the dataset
features, image_paths, genus_labels = extract_features_from_folder(unified_folder)


# In[34]:


# K-Means Clustering Application
n_clusters = 5  # 5 genuses 
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(features)


# In[35]:


# Using PCA to reduce dimension to 2D and plotting the clusters
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering of Mushroom Genus Images (PCA Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster ID')
plt.show()


# In[36]:


# Images from each cluster
cluster_images = {i: [] for i in range(n_clusters)}

for i, label in enumerate(kmeans.labels_):
    cluster_images[label].append(image_paths[i])

plt.figure(figsize=(12, 12))
for i in range(n_clusters):
    plt.subplot(n_clusters, 1, i + 1)
    random_image_path = random.choice(cluster_images[i])
    img = plt.imread(random_image_path)
    plt.imshow(img)
    plt.title(f"Cluster {i}")
    plt.axis('off')

plt.tight_layout()


# In[37]:


cluster_counts = [np.sum(kmeans.labels_ == i) for i in range(n_clusters)]

plt.figure(figsize=(10, 6))
sns.barplot(x=list(range(n_clusters)), y=cluster_counts)
plt.title('Cluster-wise Distribution of Images (K-Means)')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Images in Cluster')
plt.show()


# In[39]:


# Analyzing how genus labels are distributed across clusters
cluster_genus_distribution = {i: [] for i in range(n_clusters)}
for i, label in enumerate(kmeans.labels_):
    cluster_genus_distribution[label].append(genus_labels[i])

# Distribution of genera in each cluster
cluster_genus_counts = {i: {} for i in range(n_clusters)}
for cluster_id, genera in cluster_genus_distribution.items():
    for genus in genera:
        if genus in cluster_genus_counts[cluster_id]:
            cluster_genus_counts[cluster_id][genus] += 1
        else:
            cluster_genus_counts[cluster_id][genus] = 1

# Counts of genera per cluster
for cluster_id, counts in cluster_genus_counts.items():
    print(f"Cluster {cluster_id}:")
    for genus, count in counts.items():
        print(f"  {genus}: {count} images")
    print()


# In[42]:


# 2D PCA visualization with cluster labels and genus information
plt.figure(figsize=(14, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))  # Color map for clusters

for i in range(n_clusters):
    # Calculate the mean (centroid) of the cluster in the reduced feature space
    class_center = np.mean(reduced_features[kmeans.labels_ == i], axis=0)
    
    # Plot the centroid as a large 'x'
    plt.scatter(class_center[0], class_center[1], color=colors[i], s=200, marker='x', label=f'Cluster {i} Center')

# Plot the clusters with the PCA-reduced features
for i in range(n_clusters):
    plt.scatter(reduced_features[kmeans.labels_ == i, 0], 
                reduced_features[kmeans.labels_ == i, 1], 
                label=f'Cluster {i}', 
                color=colors[i], alpha=0.7)

plt.title(f'K-Means Clustering with PCA Dimensionality Reduction\nfor Clusters 0 through {n_clusters - 1}', fontsize=15)
plt.xlabel('Principal Component 1', fontsize=15)
plt.ylabel('Principal Component 2', fontsize=15)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Mushroom Genus", fontsize=12)
plt.tight_layout()  
plt.show()


# Supervised (or Self-Supervised) Machine Learning Models
# 
# # CNN Model
# 
# I chose a Convolutional Neural Network model to process and analyze the images. 

# In[43]:


# Splitting Data Into Training, Validation, and Testing Sets

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

unified_folder = 'unified_folder'
image_paths = []
labels = []
genus_names = os.listdir(unified_folder)

# Assign numerical labels to each genus
label_map = {genus: idx for idx, genus in enumerate(genus_names)}

for genus_name in os.listdir(unified_folder):
    genus_folder = os.path.join(unified_folder, genus_name)
    if os.path.isdir(genus_folder):
        for image_file in os.listdir(genus_folder):
            image_paths.append(os.path.join(genus_folder, image_file))
            labels.append(label_map[genus_name])

labels = to_categorical(labels)

# Split data into training (80%), validation (10%), and testing (10%) sets
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Further split the training set into training (80%) and validation (20%) sets
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=42)

# Print the number of images in each set for confirmation
print(f'Training images: {len(train_paths)}')
print(f'Validation images: {len(val_paths)}')
print(f'Test images: {len(test_paths)}')


# In[44]:


# Resizing and Normalization

# Function to load and preprocess images
def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  

# Load and preprocess the images for training, validation, and testing
train_data = np.vstack([load_and_preprocess_image(img_path) for img_path in train_paths])
val_data = np.vstack([load_and_preprocess_image(img_path) for img_path in val_paths])
test_data = np.vstack([load_and_preprocess_image(img_path) for img_path in test_paths])


# In[68]:


# CNN Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(genus_names), activation='softmax')  # Number of classes = to number of genuses
])

# Compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Summary 
model.summary()


# In[69]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the model with augmented data
history = model.fit(
    datagen.flow(train_data, train_labels, batch_size=32),
    validation_data=(val_data, val_labels),
    epochs=10,
    steps_per_epoch=len(train_data) // 32,
    validation_steps=len(val_data) // 32
)


# In[70]:


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')


# In[71]:


# Plot training & validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[73]:


# Model predictions on the test set
test_preds = model.predict(test_data)
test_preds_class = np.argmax(test_preds, axis=1)
test_labels_class = np.argmax(test_labels, axis=1)

# Accuracy for each genus
genus_accuracy = {genus: 0 for genus in genus_names}

correct_per_genus = {genus: 0 for genus in genus_names}
total_per_genus = {genus: 0 for genus in genus_names}
for i, pred in enumerate(test_preds_class):
    true_genus = genus_names[test_labels_class[i]]
    predicted_genus = genus_names[pred]

    total_per_genus[true_genus] += 1
    if true_genus == predicted_genus:
        correct_per_genus[true_genus] += 1

# Accuracy per genus
for genus in genus_names:
    genus_accuracy[genus] = correct_per_genus[genus] / total_per_genus[genus]

    
sorted_genus = sorted(genus_accuracy.items(), key=lambda item: item[1], reverse=True)

# Sorted genus names and accuracy values for plotting
genus_names_sorted = [x[0] for x in sorted_genus]
accuracy_sorted = [x[1] for x in sorted_genus]

# Test Data accuracy by genus
plt.figure(figsize=(12, 6))
plt.barh(genus_names_sorted, accuracy_sorted, color='skyblue')
plt.title('Test Data Accuracy by Genus')
plt.xlabel('Accuracy')
plt.ylabel('Genus')
plt.show()


# In[74]:


# Model predictions on the test set
test_preds = model.predict(test_data)
test_preds_class = np.argmax(test_preds, axis=1)
test_labels_class = np.argmax(test_labels, axis=1)

# Subset of random indices for visualization
num_samples = 10
random_indices = random.sample(range(len(test_preds_class)), num_samples)

plt.figure(figsize=(12, 12))
for i, idx in enumerate(random_indices):
    img_path = test_paths[idx]
    true_genus = genus_names[test_labels_class[idx]]
    predicted_genus = genus_names[test_preds_class[idx]]
    predicted_prob = test_preds[idx][test_preds_class[idx]]  
    img = image.load_img(img_path, target_size=(150, 150))
    plt.subplot(5, 2, i + 1)  
    plt.imshow(img)
    plt.axis('off')  
    plt.title(f"Actual: {true_genus}\nPredicted: {predicted_genus}\nProb: {predicted_prob:.2f}")

plt.tight_layout()
plt.show()


# In[76]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(test_labels_class, classes=np.arange(len(genus_names)))
y_pred_prob = model.predict(test_data)  
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_prob.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# # DNN Model
# Deep neural network

# In[55]:


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Flatten(input_shape=(150, 150, 3)))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(train_labels.shape[1], activation='softmax')) 
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_data, 
    train_labels, 
    epochs=50, 
    batch_size=32, 
    validation_data=(val_data, val_labels), 
    callbacks=[early_stopping]
)


# In[56]:


test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")


# In[57]:


# Training & Validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Training & Validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[58]:


from sklearn.metrics import accuracy_score
import seaborn as sns

test_predictions = model.predict(test_data)
predicted_labels = np.argmax(test_predictions, axis=1) 
true_labels = np.argmax(test_labels, axis=1)

genus_accuracy = {}

for genus_index in np.unique(true_labels):
    genus_indices = np.where(true_labels == genus_index)[0]
    genus_true_labels = true_labels[genus_indices]
    genus_predicted_labels = predicted_labels[genus_indices]
    accuracy = accuracy_score(genus_true_labels, genus_predicted_labels)
    genus_accuracy[genus_index] = accuracy

sorted_genus_accuracy = {genus_names[i]: genus_accuracy[i] for i in genus_accuracy}

plt.figure(figsize=(12, 6))
sns.barplot(x=list(sorted_genus_accuracy.keys()), y=list(sorted_genus_accuracy.values()), palette='viridis')
plt.title('Test Data Accuracy by Genus')
plt.xlabel('Genus')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[59]:


test_preds = model.predict(test_data)
test_preds_class = np.argmax(test_preds, axis=1)  
test_labels_class = np.argmax(test_labels, axis=1)  

num_samples = 10
random_indices = random.sample(range(len(test_preds_class)), num_samples)

plt.figure(figsize=(12, 12))
for i, idx in enumerate(random_indices):
    img_path = test_paths[idx]  
    true_genus = genus_names[test_labels_class[idx]]  
    predicted_genus = genus_names[test_preds_class[idx]]  
    predicted_prob = test_preds[idx][test_preds_class[idx]]  
    img = image.load_img(img_path, target_size=(150, 150))
    
    plt.subplot(5, 2, i + 1) 
    plt.imshow(img)
    plt.axis('off') 
    plt.title(f"Actual: {true_genus}\nPredicted: {predicted_genus}\nProb: {predicted_prob:.2f}")
    
plt.tight_layout()
plt.show()


# In[61]:


from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(test_labels_class, test_preds_class)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genus_names, yticklabels=genus_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# In[62]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the labels for multi-class ROC
y_test_bin = label_binarize(test_labels_class, classes=np.arange(len(genus_names)))
y_pred_bin = model.predict(test_data)

fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# ROC curve visualizes the performance of a classifier across different classification thresholds.
# 
# AUC is the overall perfomance of the model. The higher, the more accurate. 

# In[63]:


from sklearn.metrics import classification_report

# Get precision, recall, and F1 score
report = classification_report(test_labels_class, test_preds_class, target_names=genus_names)
print(report)


# # Conclusions
# When comparing the CNN and DNN model, the CNN model perfomed better. It had a higher test accuracy, and AUC. It also had a lower test loss. 
# 
# CNN
# - Test Loss: 1.5037
# - Test Accuracy: 0.3480  
# - AUC = 0.68
# 
# DNN
# - Test Loss: 1.6045
# - Test accuracy: 0.2440   
# - AUC = 0.54
# 
# This makes sense since CNNs are more efficient for image-related modeling. 
# 
# The clustering gernerally contained similar amounts from all five genuses. 
# 
# These images were dificult to differentiate. More tuning as well as different models could produce a higher accuracy. I really wish the images were clearer. Many contained background foliage. Having really exact images for training and then having in-the-field images for testing would be very interesting. Having 360 images would also be something very cool to look into. 

# In[ ]:




