import os
import yaml
import tensorflow as tf
from src.data.dataset import load_datasets
from src.models.model import build_model

# --- Load configuration ---
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

IMG_SIZE = config["data"]["img_size"]
BATCH_SIZE = config["data"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]
NUM_CLASSES = config["model"]["num_classes"]

# --- Load datasets ---
train_ds, val_ds, class_names = load_datasets(
    data_dir="data/raw",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# --- Build model ---
model = build_model(NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --- Prepare model checkpoint ---
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

# Save only the **best model** according to validation accuracy
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Optional: early stopping to avoid overfitting
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# --- Train the model ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint_callback, earlystop_callback]
)

print(f"Training finished. Best model saved at: {checkpoint_path}")
