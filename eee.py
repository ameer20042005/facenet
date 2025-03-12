import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# تهيئة Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# تحميل قاعدة بيانات الوجوه
def load_faces(folder_path):
    face_encodings = {}
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if os.path.isdir(person_folder):
            images = [os.path.join(person_folder, img) for img in os.listdir(person_folder) if
                      img.endswith(('.jpg', '.png'))]
            if images:
                face_encodings[person_name] = images[0]  # نأخذ صورة واحدة لكل شخص
    return face_encodings

# التعرف على الوجوه في الوقت الفعلي
def recognize_faces():
    face_db = load_faces("database")
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # تحويل الصورة إلى RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # اكتشاف الوجوه
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(
                        bboxC.height * h)

                    face_crop = frame[y:y + h_box, x:x + w_box]

                    # مطابقة الوجه مع قاعدة البيانات باستخدام DeepFace
                    best_match = "unknown"
                    best_score = 0.3  # عتبة التعرف على الوجه (كلما قلّت القيمة، زادت الدقة)

                    for name, img_path in face_db.items():
                        try:
                            result = DeepFace.verify(face_crop, img_path, model_name='Facenet', enforce_detection=False)
                            if result["distance"] < best_score:
                                best_score = result["distance"]
                                best_match = name
                        except:
                            pass

                    # رسم الصندوق واسم الشخص
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(frame, best_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# تشغيل البرنامج
recognize_faces()
