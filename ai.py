import cv2
import os
from deepface import DeepFace

# تحديد مسار قاعدة بيانات المستخدمين
db_path = "database"

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # البحث عن الوجه في قاعدة البيانات باستخدام نموذج دقيق
        results = DeepFace.find(
            img_path=frame,
            db_path=db_path,
            model_name="Facenet512",  # نموذج دقيق جدًا
            detector_backend="retinaface",  # كاشف وجه دقيق
            enforce_detection=False
        )

        if results and not results[0].empty:
            # استخراج المسار المطابق الأول
            identity_path = results[0]["identity"][0]

            # استخراج اسم المستخدم من اسم المجلد الأب
            identity = os.path.basename(os.path.dirname(identity_path))
            print(f"✅ تم التعرف على: {identity}")

            # الكشف عن الوجه باستخدام RetinaFace للحصول على الإحداثيات الدقيقة
            detected_faces = DeepFace.detectFace(frame, detector_backend="retinaface", enforce_detection=False, align=True)

            # التأكد من أن الوجه تم التعرف عليه
            if isinstance(detected_faces, list) and len(detected_faces) > 0:
                for face in detected_faces:
                    x, y, w, h = face["x"], face["y"], face["w"], face["h"]

                    # رسم المربع حول الوجه (أخضر)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # كتابة اسم المستخدم بلون مختلف (أزرق)
                    cv2.putText(frame, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        else:
            print("❌ لم يتم التعرف على الوجه.")

    except Exception as e:
        print("⚠️ خطأ أثناء التعرف:", e)

    # عرض الفيديو في الوقت الفعلي
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # اضغط 'q' للخروج
        break

cap.release()
cv2.destroyAllWindows()
