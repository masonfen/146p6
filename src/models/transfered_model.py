from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
import os

class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Load the base model
        results_dir = 'results'
        model_files = [f for f in os.listdir(results_dir) if f.endswith('.keras') and 'basic_model' in f]
        if not model_files:
            raise Exception("No base model found in results directory for transfer learning.")
        
        model_files.sort()
        base_model_path = os.path.join(results_dir, model_files[-1])
        print(f"Loading base model for transfer: {base_model_path}")
        base_model = models.load_model(base_model_path)
        
        # Remove the final softmax layer
        base_model.pop()
        
        # Freeze the base model weights
        base_model.trainable = False
        
        # Build the new model
        self.model = Sequential([
            base_model,
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
