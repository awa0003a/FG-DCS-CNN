import tensorflow as tf
from model import build_fg_dcs_cnn_model

# Create and compile the model
input_shape = (500, 19)  # (Time steps, Channels) - Note: channels last format
model = build_fg_dcs_cnn_model(input_shape)

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Display model architecture
model.summary()

# Optional: Visualize model architecture
tf.keras.utils.plot_model(model, to_file='fg_dcs_cnn_architecture.png', 
                          show_shapes=True, show_layer_names=True, dpi=150)