import cv2
import os
from deepface import DeepFace
import numpy as np

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
            enforce_detection=True
        )

        if results and not results[0].empty:
            # استخراج المسار المطابق الأول
            identity_path = results[0]["identity"][0]

            # استخراج اسم المستخدم من اسم المجلد الأب
            identity = os.path.basename(os.path.dirname(identity_path))
            print(f"✅ تم التعرف على: {identity}")

            # الكشف عن الوجه باستخدام RetinaFace للحصول على الإحداثيات الدقيقة
            detections = DeepFace.detectFace(frame, detector_backend="retinaface", enforce_detection=False)

            if isinstance(detections, np.ndarray):  # التأكد من وجود وجه
                h, w, _ = frame.shape

                # تحديد الإحداثيات من نسبة الصورة
                x = int(w * 0.3)
                y = int(h * 0.3)
                x2 = int(w * 0.7)
                y2 = int(h * 0.7)

                # رسم المربع حول الوجه باللون الأخضر
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # عرض الاسم بلون أزرق مختلف عن لون الإطار
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
