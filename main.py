import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# Configurações gerais
DATASET_DIR = '/'  # Substituir pelo caminho real do dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

from PIL import UnidentifiedImageError

def safe_load_img(path):
    try:
        return tf.keras.preprocessing.image.load_img(path)
    except UnidentifiedImageError:
        print(f"Erro ao carregar a imagem: {path}")
        return None

# Modifique o fluxo de dados para usar a função safe_load_img
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    preprocessing_function=safe_load_img
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Definição do modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compilação do modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinamento do modelo
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Função para análise ao vivo da câmera
def analyze_camera_feed():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o frame.")
            break

        # Redimensiona e processa o frame
        resized_frame = cv2.resize(frame, IMG_SIZE)
        img_array = resized_frame / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predição do modelo
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]
        class_name = list(train_data.class_indices.keys())[class_idx]

        # Exibe o resultado
        cv2.putText(
            frame,
            f"Estilo: {class_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Função principal
if __name__ == "__main__":
    print("Treinamento concluído. Iniciando análise da câmera...")
    analyze_camera_feed()
