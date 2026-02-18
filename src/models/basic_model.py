from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Optimized architecture for Section 6 (>70% accuracy, <150k params)
        self.model = Sequential([
            layers.Input(shape=input_shape),
            
            # Data Augmentation to prevent overfitting and improve generalization
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            
            Rescaling(1./255),
            
            # 5 Convolutional layers with pooling to reduce spatial dimensions to 4x4
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)), # 75x75
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)), # 37x37
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)), # 18x18
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)), # 9x9
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)), # 4x4
            
            layers.Flatten(), # 4 * 4 * 64 = 1024 neurons
            
            # Dense layers tuned to stay under 150k total parameters
            layers.Dense(48, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        # Switching to Adam optimizer for better convergence
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
