from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
import os
import numpy as np

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Load the base model to mirror its architecture exactly
        results_dir = 'results'
        model_files = [f for f in os.listdir(results_dir) if f.endswith('.keras') and 'basic_model' in f]
        if not model_files:
            raise Exception("No base model found in results directory for architecture mirroring.")
        
        model_files.sort()
        base_model_path = os.path.join(results_dir, model_files[-1])
        base_model = models.load_model(base_model_path)
        
        # Remove the final softmax layer
        base_model.pop()
        
        # Build the new model
        self.model = Sequential([
            base_model,
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(categories_count, activation='softmax')
        ])
        
        # Randomize all weights in the model
        self._randomize_layers(self.model)
        
        # For the "without transfer" model, all parameters should be available for learning
        self.model.trainable = True
    
    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    @staticmethod
    def _randomize_layers(model):
        for layer in model.layers:
            # Handle nested Sequential models (like the base_model we bolted on)
            if hasattr(layer, 'layers'):
                RandomModel._randomize_layers(layer)
            
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                if weights:
                    new_weights = []
                    for w in weights:
                        if len(w.shape) > 1: # Kernel
                            new_weights.append(layer.kernel_initializer(shape=w.shape).numpy())
                        else: # Bias
                            new_weights.append(layer.bias_initializer(shape=w.shape).numpy())
                    layer.set_weights(new_weights)
