import tensorflow as tf
from model import build_fg_dcs_cnn_model

def train_model(model, train_x, train_y, val_x, val_y, epochs=50, batch_size=32):
    """
    Training function with callbacks for better performance monitoring
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint('best_fg_dcs_cnn.h5', save_best_only=True)
    ]
    
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Example usage:
# Ensure your data is in the correct format: (samples, time_steps, channels)
# from main import model  # Assuming model is built in main.py
# history = train_model(model, train_x, train_y, val_x, val_y)