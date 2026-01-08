import cv2
import mediapipe as mp
import csv
import os

class HandDataCollector:
    def __init__(self, filename="data_hand.csv"):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # UBAH 1: max_num_hands jadi 2
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.filename = filename
        self.init_csv()

    def init_csv(self):
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                # UBAH 2: Header disiapkan untuk 84 fitur (42 titik x,y)
                # Walaupun tangan cuma 1, kolom tetap harus disediakan
                header = ['label']
                for i in range(42): # 21 titik tangan kiri + 21 titik tangan kanan
                    header.extend([f'x{i}', f'y{i}'])
                writer.writerow(header)

    def detect_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        
        return image, results

    def save_data(self, key_pressed, results):
        if results.multi_hand_landmarks:
            # Siapkan baris data
            data_row = []
            
            # Ambil semua titik dari semua tangan yang terdeteksi
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_row.append(landmark.x)
                    data_row.append(landmark.y)
            
            # UBAH 3: Logika Padding (Isi Kekosongan)
            # Target kita adalah 84 data (21 titik * 2 koor * 2 tangan)
            expected_features = 84 
            
            # Jika data kurang (misal cuma 1 tangan = 42 data), sisanya diisi 0
            if len(data_row) < expected_features:
                sisanya = expected_features - len(data_row)
                data_row.extend([0.0] * sisanya)
            
            # Potong jika berlebih (jaga-jaga error)
            data_row = data_row[:expected_features]

            # Masukkan label huruf di depan
            final_row = [chr(key_pressed).upper()] + data_row

            # Simpan ke CSV
            with open(self.filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(final_row)
            
            print(f"Data tersimpan untuk huruf: {chr(key_pressed).upper()} (2 Tangan Support)")
            return True
        return False