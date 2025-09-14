# FG-DCS-CNN: Advanced EEG Signal Classification Model

A sophisticated deep learning model for EEG-based classification, integrating **Dynamic Channel Selection (DCS)**, **Channel Attention Module (CAM)**, and **Feature Generation Module (FGM)** with a Convolutional Neural Network (CNN) backbone. This implementation follows the methodology from research on addressing class imbalance in EEG signals through synthetic feature augmentation and adaptive channel weighting. Designed for 4-class classification tasks (e.g., motor imagery or seizure detection).

## 🚀 Features
- **Dynamic Channel Selection (DCS)**: Assigns learnable importance weights to EEG channels to focus on relevant signals.
- **Channel Attention Module (CAM)**: Uses dual pooling (max + average) for enhanced feature refinement.
- **Feature Generation Module (FGM)**: Generates synthetic features to mitigate class imbalance.
- **CNN Backbone**: Multi-layer convolutions with batch normalization, pooling, and dropout for robust feature extraction.
- **End-to-End Training**: Includes early stopping, learning rate reduction, and model checkpointing for optimal performance.
- **Modular Design**: Easily extensible for custom input shapes or additional layers.

## 📊 Model Architecture Overview
The model processes EEG signals in channels-last format `(time_steps, channels)`. Key flow:
1. **Input**: Raw EEG signals with shape `(T, C)` (e.g., `(500, 19)`).
2. **DCS**: Applies importance weights to channels using global average pooling and dense layers.
3. **CAM**: Refines features by combining max and average pooling with sigmoid attention.
4. **FGM**: Generates synthetic features to augment the input, addressing class imbalance.
5. **Augmentation**: Concatenates processed and synthetic features for richer representations.
6. **CNN**: Applies multiple convolutional layers with batch normalization, max pooling, and dropout.
7. **Classification**: Uses dense layers culminating in a softmax output for 4-class prediction.

Run `python main.py` to print a detailed model summary to the console.

## 🛠️ Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fg-dcs-cnn.git
   cd fg-dcs-cnn
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install tensorflow>=2.10.0
   pip install matplotlib pydot graphviz  # For model visualization
   ```
   - **Note**: Graphviz requires system installation (e.g., `brew install graphviz` on macOS, `apt install graphviz` on Ubuntu).

## 📖 Usage
### 1. Build and Visualize Model
Run the main script to instantiate the model and generate architecture diagrams:
```
python main.py
```
- **Outputs**: 
  - Model summary printed to the console.
  - Architecture diagram saved as `fg_dcs_cnn_architecture.png`.

### 2. Train the Model
Prepare your data as NumPy arrays:
- `train_x`: Shape `(n_samples, 500, 19)` (float32).
- `train_y`: Shape `(n_samples,)` (integer labels 0-3).
- Similarly for `val_x`, `val_y`.

Import and train:
```python
from main import model  # Or rebuild in your script
from train import train_model
import numpy as np  # Assuming data loaded as np arrays

# Example: Load your data here (e.g., from .npy files)
# train_x = np.load('train_x.npy')
# train_y = np.load('train_y.npy')
# val_x = np.load('val_x.npy')
# val_y = np.load('val_y.npy')

history = train_model(model, train_x, train_y, val_x, val_y, epochs=50, batch_size=32)

# Plot training history (optional)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.savefig('training_history.png')
plt.show()
```
- **Outputs**:
  - Saves best model weights as `best_fg_dcs_cnn.h5`.
  - Callbacks include early stopping (patience=10), learning rate reduction (factor=0.5), and model checkpointing.

### 3. Inference/Prediction
Use a trained model for predictions:
```python
# Load saved model
loaded_model = tf.keras.models.load_model('best_fg_dcs_cnn.h5')

# Predict on new data (shape: (n_samples, 500, 19))
predictions = loaded_model.predict(test_x)
classes = np.argmax(predictions, axis=1)
```

### 4. Customization
- **Input Shape**: Modify `input_shape` in `main.py` (e.g., for different sampling rates or channels).
- **Output Classes**: Adjust the final `Dense` layer in `model.py` for different classification tasks.
- **Hyperparameters**: Tune convolutional filters, dense units, or dropout rates in `build_fg_dcs_cnn_model`.

## 🔬 Performance Notes
- **Loss/Optimizer**: Uses sparse categorical cross-entropy with the Adam optimizer (default learning rate=0.001).
- **Metrics**: Tracks accuracy, precision, and recall (macro-averaged).
- **Expected Performance**: On balanced EEG datasets, the model typically achieves competitive accuracy after 20-30 epochs, depending on the dataset.
- **Hardware**: Trains efficiently on CPU or GPU; configure `tf.config` for GPU acceleration if available.

## 📂 Repository Structure
```
fg-dcs-cnn/
├── model.py          # Core model definitions (DCS, CAM, FGM, build function)
├── main.py           # Model building, compilation, summary, and visualization
├── train.py          # Training function and usage example
├── README.md         # This file
└── fg_dcs_cnn_architecture.png  # Auto-generated diagram (add to .gitignore if regenerating)
```

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Suggested contributions:
- Data loaders for common EEG datasets (e.g., BCI Competition IV).
- Scripts for hyperparameter tuning.
- Additional evaluation metrics (e.g., confusion matrix, F1-score).

## 📄 License
MIT License (add a `LICENSE` file to the repository for specifics).

## 🙏 Acknowledgments
Inspired by research on EEG classification, channel attention mechanisms, and class imbalance handling. Built with TensorFlow and Keras.

For questions or issues, please open a GitHub issue!
