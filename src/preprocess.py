from tensorflow.keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split
import os

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    transfer_data_dir = 'transfer_data'
    if not os.path.exists(transfer_data_dir):
        print(f"Warning: '{transfer_data_dir}' directory not found. Please create it and add your transfer learning dataset.")
        return None, None, None

    train_dir = os.path.join(transfer_data_dir, 'train')
    test_dir = os.path.join(transfer_data_dir, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
         # Fallback if there's no train/test split, just use the root and let Keras handle validation split if possible, 
         # but for this assignment structure, we expect train/test or similar.
         # Let's assume the user puts their data in transfer_data/train and transfer_data/test similar to the main dataset.
         print(f"Warning: '{train_dir}' or '{test_dir}' not found inside '{transfer_data_dir}'. Expecting 'train' and 'test' subdirectories.")
         return None, None, None

    print('transfer train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    
    print('transfer test dataset:')
    test_dataset = image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
    
    return train_dataset, validation_dataset, test_dataset
