import tensorflow as tf

def load_datasets(data_dir, img_size, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    normalization = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    return train_ds, val_ds
