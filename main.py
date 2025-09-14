import numpy as np
from train import train_model

if __name__ == "__main__":
    # (Read the already created .npy files from the respective dataset repository).
    train_x = __
    train_y = __ 
    test_x = __
    test_y = __

    # Train the model
    model, history = train_model(
        train_x, train_y, test_x, test_y,
        input_shape=(19, 500),
        num_classes=4,
        epochs=10,
        batch_size=64
    )

    # Save trained model
    model.save("cnn_feature_gen_dcs.h5")
    print("Model training complete and saved.")
