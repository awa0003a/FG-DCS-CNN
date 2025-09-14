import tensorflow as tf
from tensorflow.keras import layers, Model

# ------------------------------
# Dynamic Channel Selection
# ------------------------------
def dynamic_channel_selection(inputs):
    num_channels = inputs.shape[-1]

    avg_pool = layers.GlobalAveragePooling1D()(inputs)
    dense1 = layers.Dense(num_channels // 2, activation='relu')(avg_pool)
    dense2 = layers.Dense(num_channels, activation='sigmoid')(dense1)

    channel_weights = layers.Reshape((1, num_channels))(dense2)
    weighted_inputs = layers.Multiply()([inputs, channel_weights])

    return weighted_inputs


# ------------------------------
# Feature Generation Module
# ------------------------------
class FeatureGenerationModule(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(FeatureGenerationModule, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.dense1 = layers.Dense(input_shape[-1], activation='relu')
        self.dense2 = layers.Dense(self.output_dim, activation='linear')
        super(FeatureGenerationModule, self).build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


# ------------------------------
# Channel Attention Module
# ------------------------------
def channel_attention_module(inputs):
    channels = inputs.shape[-1]
    avg_pool = layers.GlobalAveragePooling1D()(inputs)
    dense1 = layers.Dense(channels // 8, activation='relu')(avg_pool)
    dense2 = layers.Dense(channels, activation='sigmoid')(dense1)
    scale = layers.Reshape((1, channels))(dense2)
    return layers.Multiply()([inputs, scale])


# ------------------------------
# Build Model
# ------------------------------
def build_model_with_feature_generation(input_shape, num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)

    # Feature Generation
    feature_gen = FeatureGenerationModule(output_dim=input_shape[-1])(inputs)
    augmented_inputs = layers.Concatenate()([inputs, feature_gen])

    # Dynamic Channel Selection
    dcs_out = dynamic_channel_selection(augmented_inputs)

    # First Convolutional Block
    conv1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(dcs_out)
    conv1 = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)

    # Channel Attention
    channel_attention = channel_attention_module(conv1)

    # Second Convolutional Block
    conv2 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(channel_attention)
    conv2 = layers.MaxPooling1D(pool_size=2)(conv2)
    conv2 = layers.Dropout(0.3)(conv2)

    # Fully Connected Layers
    flatten = layers.Flatten()(conv2)
    dense1 = layers.Dense(64, activation='relu')(flatten)
    dense1 = layers.Dropout(0.5)(dense1)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(dense1)

    return Model(inputs, outputs)
