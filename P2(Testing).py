# ===========================================================
# Step 5: Model Testing
# ===========================================================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Define paths for test images
test_images = {
    "crack": "P2Data/test/crack/test_crack.jpg",
    "missing-head": "P2Data/test/missing-head/test_missinghead.jpg",
    "paint-off": "P2Data/test/paint-off/test_paintoff.jpg"
}

# Class labels
class_labels = ["crack", "missing-head", "paint-off"]
IMG_SIZE = (500, 500)

# Preprocess and Predict Function
def preprocess_and_predict(model, img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis = 0)
    preds = model.predict(img_array)
    return img, preds[0]

# Display Predictions Function
def display_predictions(model_name, model_path, save_name):
    print(f"\n✅ {model_name} loaded successfully!")
    model = keras.models.load_model(model_path)

    plt.figure(figsize=(12, 4))
    for i, (true_label, path) in enumerate(test_images.items()):
        img, pred_probs = preprocess_and_predict(model, path)
        pred_index = np.argmax(pred_probs)
        pred_label = class_labels[pred_index]
        confidence = np.max(pred_probs)

        prediction_text = "\n".join([
            f"{lbl}: {prob*100:.1f}%"
            for lbl, prob in zip(class_labels, pred_probs)
        ])

        # Plot Image
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")

        # Color-Code Title: green if correct, red if wrong
        color = "green" if true_label == pred_label else "red"
        plt.title(f"Actual: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)",
                  color = color, fontsize = 10
)

        plt.text(
            5, IMG_SIZE[0] - 80, prediction_text,
            fontsize = 9, color = 'white', backgroundcolor = 'black',
            ha='left', va='top', family='monospace'
        )

    plt.suptitle(f"{model_name} - Predictions on Test Images", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

# Run for both models
display_predictions("Model 1", "model1_aircraft_defect_cnn.keras", "Model1_Predictions.png")
display_predictions("Model 2", "model2_aircraft_defect_cnn.keras", "Model2_Predictions.png")