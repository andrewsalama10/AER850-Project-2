# ===========================================================
# Step 1: Data Processing - 20 Marks
# ===========================================================
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For Reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)

# Image Shape, Image Size, and Batch Size
IMG_SHAPE = (500, 500, 3)
IMG_SIZE = (500, 500)
BATCH_SIZE = 32

# Relative Paths
train_dir = "P2Data/train"
val_dir   = "P2Data/valid"
test_dir  = "P2Data/test"

# Training Data Augementation
train_datagen = ImageDataGenerator(
    rescale = 1.0/255.0,
    shear_range = 0.3,
    zoom_range = 0.2,
    rotation_range = 25,
    horizontal_flip = True,
    vertical_flip = True
)

# Rescaling Validation and Testing Data
val_datagen = ImageDataGenerator(
    rescale = 1.0/255.0
)

test_datagen = ImageDataGenerator(
    rescale = 1.0/255.0
)  

# Create Generators
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    shuffle = True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    shuffle = False
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    shuffle = True
)

print("Part 1: Data Processing is Succesfull")

# ======================================================================
# Step 2: Neural Network Architecture Design (DCNN Model) - 30 Marks
# ======================================================================
from tensorflow.keras import layers, Input

# Create Nerual Network Model
# MODEL 1
mdl = keras.Sequential([
    Input(shape = IMG_SHAPE),
    layers.Conv2D(32, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(256, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(256, activation = 'relu'),
    layers.Dropout(0.6),
    layers.Dense(3, activation = 'softmax', dtype = 'float32')
])

# MODEL 2
mdl_v2 = keras.Sequential([
    Input(shape = IMG_SHAPE),

    layers.Conv2D(32, (3,3), activation = 'elu'),
    layers.ELU(alpha = 0.1),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation = 'elu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation = 'elu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation = 'elu'),
    layers.Dense(64, activation = 'elu'),
    layers.Dropout(0.5),

    layers.Dense(3, activation='softmax', dtype='float32')
])

print("Part 2: NNA (DCNN Models) is Sucessfull")

# ===========================================================
# Step 3: Hyperparameter Analysis - 20 Marks
# ===========================================================
# Compile Model
# MODEL 1
mdl.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

mdl.summary()

# MODEL 2
mdl_v2.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 5e-5),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

print("Part 3: Hyperparameters is Sucessfull")

# ===========================================================
# Step 4: Training and Evaluation - 10 Marks
# ===========================================================
import matplotlib.pyplot as plt

# MODEL 1
# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy',
    patience = 5,
    restore_best_weights = True
)

EPOCHS = 35

history = mdl.fit(
    train_gen,
    validation_data = val_gen,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = [early_stop],
    verbose = 1
)

# Save model
mdl.save("model1_aircraft_defect_cnn.keras")

# Plot Model Accuracy vs EPOCH
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model 1 Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("Model 1 Accuracy.png")
plt.close()

# Plot Model Loss vs EPOCH
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 1 Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("Model 1 Loss.png")
plt.close()

# Evaluate model on all test images
test_loss, test_acc = mdl.evaluate(test_gen, verbose = 1)

print(f"✅ Model 1 Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# MODEl 2
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience = 7,
    restore_best_weights = True
)

EPOCHS = 40

history_v2 = mdl_v2.fit(
    train_gen,
    validation_data = val_gen,
    epochs=EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = [early_stop],
    verbose = 1
)

# Save model
mdl_v2.save("model2_aircraft_defect_cnn.keras")

# Accuracy and Loss Plots
plt.figure(figsize = (6,4))
plt.plot(history_v2.history["accuracy"], label = "Training Accuracy")
plt.plot(history_v2.history["val_accuracy"], label = "Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model 2 Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("Model2_Accuracy.png")
plt.close()

plt.figure(figsize = (6,4))
plt.plot(history_v2.history["loss"], label = "Training Loss")
plt.plot(history_v2.history["val_loss"], label = "Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 2 Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("Model2_Loss.png")
plt.close()

# Model Evaluation on Test Data
test_loss_v2, test_acc_v2 = mdl_v2.evaluate(test_gen, verbose = 1)
print(f"✅ Model 2 Test Accuracy: {test_acc_v2:.4f} | Test Loss: {test_loss_v2:.4f}")
