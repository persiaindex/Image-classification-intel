import tensorflow as tf
from sklearn.metrics import classification_report

model = tf.keras.models.load_model("model.h5")

# Extend this later with test set + confusion matrix
print("Model loaded. Ready for evaluation.")
