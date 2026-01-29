import yaml
import tensorflow as tf
from src.data.dataset import load_datasets
from src.models.model import build_model

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

train_ds, val_ds = load_datasets(
    data_dir="data/raw",
    img_size=config["data"]["img_size"],
    batch_size=config["data"]["batch_size"]
)

model = build_model(config["model"]["num_classes"])

model.compile(
    optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["training"]["epochs"]
)

model.save("model.h5")
