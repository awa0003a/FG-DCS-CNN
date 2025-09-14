import tensorflow as tf
from tensorflow.keras import layers, models, Model

def dynamic_channel_selection(inputs):
    """
    Dynamic Channel Selection Module - Applied to raw EEG input
    Assigns learnable importance weights to channels
    """
    # Global Average Pooling: X̄_c = (1/T) Σ X_t,c
    avg_pool = layers.GlobalAveragePooling1D()(inputs)
    
    # First Dense Layer: h = ReLU(W₁X̄ + b₁)
    dense1 = layers.Dense(inputs.shape[-1] // 2, activation='relu', name='dcs_dense1')(avg_pool)
    
    # Second Dense Layer: α = σ(W₂h + b₂)
    dense2 = layers.Dense(inputs.shape[-1], activation='sigmoid', name='dcs_dense2')(dense1)
    
    # Reshape for broadcasting: α ∈ R^C → α ∈ R^(1×C)
    channel_weights = layers.Reshape((1, inputs.shape[-1]), name='dcs_weights')(dense2)
    
    # Apply channel weights: X̂_t,c = X_t,c · α_c
    weighted_inputs = layers.Multiply(name='dcs_apply_weights')([inputs, channel_weights])
    
    return weighted_inputs

def channel_attention_module(inputs):
    """
    Channel Attention Module - Applied after DCS
    Uses dual pooling (max + average) for comprehensive channel assessment
    """
    # Global Max Pooling: Max_c = max_t(X̂_t,c)
    max_pool = layers.GlobalMaxPooling1D()(inputs)
    
    # Global Average Pooling: Avg_c = (1/T) Σ X̂_t,c
    avg_pool = layers.GlobalAveragePooling1D()(inputs)
    
    # Feature Fusion: P_c = Max_c + Avg_c
    combined = layers.Add(name='cam_feature_fusion')([max_pool, avg_pool])
    
    # Attention Weight Generation: β_c = σ(W_att^T P + b_c)
    attention_weights = layers.Dense(inputs.shape[-1], activation='sigmoid', name='cam_attention')(combined)
    
    # Reshape for broadcasting: β ∈ R^C → β ∈ R^(1×C)
    attention_weights = layers.Reshape((1, inputs.shape[-1]), name='cam_reshape')(attention_weights)
    
    # Apply attention: X̃_t,c = X̂_t,c · β_c
    attended_output = layers.Multiply(name='cam_apply_attention')([inputs, attention_weights])
    
    return attended_output

class FeatureGenerationModule(layers.Layer):
    """
    Feature Generation Module - Creates synthetic features to address class imbalance
    Applied after channel selection and attention
    """
    def __init__(self, output_dim, **kwargs):
        super(FeatureGenerationModule, self).__init__(**kwargs)
        self.output_dim = output_dim
        
    def build(self, input_shape):
        # First Dense Layer: h_t,c = ReLU(W₁ · X_t,c + b₁)
        self.dense1 = layers.Dense(input_shape[-1], activation='relu', name='fgm_dense1')
        
        # Second Dense Layer: z_t,d = W₂h_t,c + b₂
        self.dense2 = layers.Dense(self.output_dim, activation='linear', name='fgm_dense2')
        
        super(FeatureGenerationModule, self).build(input_shape)
    
    def call(self, inputs):
        # Generate synthetic features Z ∈ R^(T×D)
        h = self.dense1(inputs)
        synthetic_features = self.dense2(h)
        return synthetic_features

def build_fg_dcs_cnn_model(input_shape):
    """
    Complete FG-DCS-CNN Model following paper methodology:
    EEG Input → DCS → CAM → FGM → CNN → Classification
    """
    inputs = tf.keras.Input(shape=input_shape, name='eeg_input')
    
    # Step 1: Dynamic Channel Selection
    # Input: X ∈ R^(T×C) → Output: X̂ ∈ R^(T×C)
    dcs_output = dynamic_channel_selection(inputs)
    
    # Step 2: Channel Attention Module
    # Input: X̂ ∈ R^(T×C) → Output: X̃ ∈ R^(T×C)
    cam_output = channel_attention_module(dcs_output)
    
    # Step 3: Feature Generation Module
    # Generate synthetic features from processed signals
    feature_gen = FeatureGenerationModule(output_dim=input_shape[-1], name='feature_generation')
    synthetic_features = feature_gen(cam_output)
    
    # Concatenate original processed features with synthetic features
    # X̃ ∈ R^(T×C) + Z ∈ R^(T×D) → Augmented ∈ R^(T×(C+D))
    augmented_inputs = layers.Concatenate(axis=-1, name='feature_concatenation')([cam_output, synthetic_features])
    
    # Step 4: CNN Processing
    # First Convolutional Layer
    conv1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1')(augmented_inputs)
    conv1 = layers.BatchNormalization(name='bn1')(conv1)
    
    # Second Convolutional Layer
    conv2 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv2')(conv1)
    conv2 = layers.MaxPooling1D(pool_size=2, name='maxpool1')(conv2)
    conv2 = layers.Dropout(0.3, name='dropout1')(conv2)
    
    # Additional Convolutional Layer
    conv3 = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv3')(conv2)
    conv3 = layers.MaxPooling1D(pool_size=2, name='maxpool2')(conv3)
    conv3 = layers.Dropout(0.3, name='dropout2')(conv3)
    
    # Flatten for Fully Connected Layers
    flatten = layers.Flatten(name='flatten')(conv3)
    
    # Dense Layers
    dense1 = layers.Dense(128, activation='relu', name='dense1')(flatten)
    dense1 = layers.Dropout(0.5, name='dropout3')(dense1)
    
    dense2 = layers.Dense(64, activation='relu', name='dense2')(dense1)
    dense2 = layers.Dropout(0.5, name='dropout4')(dense2)
    
    # Output Layer for 4-class Classification
    outputs = layers.Dense(4, activation='softmax', name='output')(dense2)
    
    return Model(inputs=inputs, outputs=outputs, name='FG_DCS_CNN')