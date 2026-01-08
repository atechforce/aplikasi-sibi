import cv2
import mediapipe as mp
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. Load Model
try:
    model_dict = pickle.load(open('./model_sibi.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: File model_sibi.p tidak ditemukan.")
    exit()

# 2. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Support 2 tangan
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# 3. Nyalakan Kamera
cap = cv2.VideoCapture(0)

print("Kamera Siap (Mode 2 Tangan)! Tekan 'q' untuk keluar.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Loop semua tangan untuk digambar dan diambil datanya
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # LOGIKA BARU: Padding agar cocok dengan model 2 tangan
        expected_features = 84
        if len(data_aux) < expected_features:
            sisanya = expected_features - len(data_aux)
            data_aux.extend([0.0] * sisanya)
        
        # Potong jika lebih (misal noise)
        data_aux = data_aux[:expected_features]

        # Prediksi
        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            
        except Exception as e:
            pass

    cv2.imshow('Penerjemah SIBI (2 Tangan)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()