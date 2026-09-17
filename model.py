"""
model.py
--------
Two model options:

1. build_custom_cnn()   - a CNN built from scratch. Good for learning
                           and works fine on a few thousand images.
2. build_transfer_model() - EfficientNetB0 with ImageNet weights,
                           fine-tuned. Usually gives higher accuracy
                           with less data and less training time.
                           Recommended for the actual deliverable.

Both output a single sigmoid unit: P(tumor). Confidence for the
predicted class = max(p, 1-p).
"""

import tensorflow as tf
from tensorflow.keras import layers, models


IMG_SIZE = (224, 224)


def build_custom_cnn(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)

    # Light data augmentation baked into the model so it's applied
    # automatically during training and skipped automatically at
    # inference time (Keras handles this via the `training` flag).
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="tumor_probability")(x)

    model = models.Model(inputs, outputs, name="brain_tumor_custom_cnn")
    return model


def build_transfer_model(input_shape=(224, 224, 3), fine_tune_at=100):
    """
    EfficientNetB0 backbone pretrained on ImageNet, with a small
    classification head on top. `fine_tune_at` controls how many of
    the base model's layers stay frozen (earlier layers = more
    general features, kept frozen; later layers get fine-tuned).
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # EfficientNet has its own preprocessing built in when using
    # tf.keras.applications.efficientnet.preprocess_input on raw
    # 0-255 images, so we do NOT manually rescale here.
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="tumor_probability")(x)

    model = models.Model(inputs, outputs, name="brain_tumor_efficientnet")
    return model


def compile_model(model, learning_rate=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
