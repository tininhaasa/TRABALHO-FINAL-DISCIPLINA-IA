import os
import csv
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Função para baixar imagens e organizá-las em pastas por tipo
def download_and_organize_images(csv_file, output_dir):
    """
    Lê o arquivo CSV, baixa imagens e organiza por tipo de arte.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dicionário para contar o número de amostras por tipo de arte
    art_type_count = {art_type: 0 for art_type in ['Impressionism', 'Cubism', 'Renaissance', 'Surrealism', 'Dada', 'Expressionism']}

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            art_type = row['Style']
            
            # Filtrar para tipos de arte específicos e limitar a 500 amostras por tipo
            if art_type not in art_type_count or art_type_count[art_type] >= 200:
                continue

            url = row['Link']
            art_name = row['Artwork'].replace(' ', '_').replace(',', '').replace('"', '')

            art_dir = os.path.join(output_dir, art_type)
            if not os.path.exists(art_dir):
                os.makedirs(art_dir)

            try:
                response = requests.get(url)
                response.raise_for_status()

                # Salvar imagem
                img = Image.open(BytesIO(response.content))
                img_path = os.path.join(art_dir, f"{art_name}.jpeg")
                img.convert('RGB').save(img_path, 'JPEG')
                print(f"Imagem salva: {img_path}")

                # Incrementar o contador para o tipo de arte
                art_type_count[art_type] += 1

            except Exception as e:
                print(f"Erro ao baixar ou salvar {url}: {e}")

# Função para verificar imagens inválidas e removê-las
def verify_images(directory):
    """
    Verifica todas as imagens em um diretório e subdiretórios.
    Remove arquivos que não são imagens válidas.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img.verify()  # Verifica se a imagem é válida
            except (IOError, SyntaxError) as e:
                print(f"Imagem inválida removida: {img_path}")
                os.remove(img_path)

# CASO PRECISE BAIXAR AS IMAGENS NOVAMENTE, DESCOMENTE AS LINHAS ABAIXO
# download_and_organize_images('wikiart.csv', 'output_directory')
# verify_images('output_directory')

# Verifique se o diretório contém imagens
for root, dirs, files in os.walk('output_directory'):
    for d in dirs:
        print(f"Diretório: {os.path.join(root, d)}, Número de imagens: {len(os.listdir(os.path.join(root, d)))}")

# Configuração do ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'output_directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'output_directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Defina o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treine o modelo
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
# Salve o modelo treinado
model.save('art_style_classifier.h5')