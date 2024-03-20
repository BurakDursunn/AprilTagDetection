import cv2
import apriltag
import numpy as np

def calculate_distance(corners):
    # Bir tag'ın köşelerinin koordinatlarını al
    corner1 = corners[0]
    corner2 = corners[1]
    corner3 = corners[2]
    corner4 = corners[3]

    # Tag'ın iki köşesi arasındaki uzunluğu hesapla
    side_length = np.linalg.norm(corner1 - corner2)

    # Tag'ın boyutunu tanımla (örneğin, bir kenarının uzunluğu 16 mm ise)
    tag_size_mm = 16  # mm

    # Raspberry Pi kamera modülünün odak uzaklığı (mm cinsinden)
    focal_length_mm = 3.04  # Raspberry Pi kamera modülünün odak uzaklığı (mm cinsinden)

    # Tag'a olan uzaklığı hesapla
    distance_mm = (focal_length_mm * tag_size_mm) / side_length

    # mm cinsinden olan mesafeyi cm cinsine çevir
    distance_cm = distance_mm / 10

    #Öylesine 300 ile çarptım rasberry pi çekince düzelt!!
    return distance_cm*300

def detect_april_tags(video_path):
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)
    # April taglarını tanımak için detektörü oluştur
    detector = apriltag.Detector()

    while True:
        # Bir frame al
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü griye dönüştür
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # April taglarını tespit et
        detections = detector.detect(gray)

        # Tespit edilen April taglarını işaretle
        for detection in detections:
            for pt in detection.corners.astype(int):
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

            # Tag'ın üst kısmına ID'nin 10 katı büyüklüğünde konum değeri ekle
            cv2.putText(frame, str(detection.tag_id * 1.5), (detection.corners[0, 0].astype(int) + 10, detection.corners[0, 1].astype(int) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Mesafeyi üst kısma ekle
            distance = calculate_distance(detection.corners)
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (detection.corners[0, 0].astype(int) + 10, detection.corners[0, 1].astype(int) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Yeşil çerçeve ve ID'yi çiz
        for detection in detections:
            cv2.polylines(frame, [detection.corners.astype(int)], True, (0, 255, 0), thickness=2)
            cv2.putText(frame, str(detection.tag_id), (detection.corners[0, 0].astype(int) + 10, detection.corners[0, 1].astype(int) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow("April Tags", frame)

        # Çıkış için 'q' tuşuna basılmasını bekle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Videoyu serbest bırak ve pencereleri kapat
    cap.release()
    cv2.destroyAllWindows()

# Video dosyası yolu
video_path = "./dörd.mp4"

# April tagları tespit et
detect_april_tags(video_path)
