import cv2
from data_collector import HandDataCollector

def main():
    #1. setup
    cap = cv2.VideoCapture(0)
    collector = HandDataCollector(filename="data_hand.csv")

    print("===PROGRAM PEREKAM DATA SIBI===")
    print("Cara pakai")
    print("1. Arahkan tangan ke kamera.")
    print("2. Tekan tombol huruf di keyboard (misal 'a') untuk merekam pose terssebut sebagai 'A'.")
    print("3. Tekan ESC untuk keluar")

    while True:
        success, frame = cap.read()
        if not success:
            break

        #flip kamera agar seperti cermin (opsi aja ini biar nda bingung)
        frame = cv2.flip(frame, 1)

        #2 deteksi tangan
        frame, results = collector.detect_hands(frame)

        #3 tampilkan pesan di layar
        cv2.putText(frame, "Tekan A-Z untuk rekam data", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 3)
        
        cv2.imshow('Data Collection SIBI', frame)

        #4 input handler
        key = cv2.waitKey(1)

        #jika tombol sc (27) ditekan, keluar

        if key == 27:
            break

        # jika tombol a-z ditekan (kode ASCII 97-122), simpan data
        elif 97 <= key <= 122:
            saved = collector.save_data(key, results)
            if saved:
                #efek visual kedip outih sebentar jika berhasil simpan
                cv2.putText(frame, "TERSIMPAN!", (10, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Data Collection SIBI', frame)
                cv2.waitKey(100) #pause dikit biar keliatan

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()       
    
