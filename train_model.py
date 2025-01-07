import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build the model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(dataset_path, save_path):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )

    model = build_model()
    model.fit(train_data, validation_data=val_data, epochs=25)
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Train and save the model
if __name__ == '__main__':
    train_model('path_to_dataset', 'emotion_model.h5')
