import tensorflow as tf
from model import build_model_with_feature_generation

def train_model(train_x, train_y, test_x, test_y,
                input_shape=(19, 500), num_classes=4,
                epochs=50, batch_size=128):
    """
    Train the model with given data.
    """
    model = build_model_with_feature_generation(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_x, train_y,
        validation_data=(test_x, test_y),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, history
