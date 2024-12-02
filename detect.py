import cv2
import tensorflow as tf
import numpy as np

# Carregar modelo treinado
model = tf.keras.models.load_model('art_style_classifier.h5')
class_names = ['Impressionism', 'Cubism', 'Renaissance', 'Surrealism', 'Dada', 'Expressionism']  # Exemplos

# Inicializar câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessar quadro
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0

    # Previsão
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    # Exibir resultados
    text = f"{class_names[class_index]} - {confidence*100:.2f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Detecção de Estilo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
