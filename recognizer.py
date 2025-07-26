import cv2
import numpy as np
import os
import time

# تحميل النموذج المدرب
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_path = "trainer/trainer.yml"

if not os.path.exists(trainer_path):
    print("\n [ERROR] ملف التدريب غير موجود!")
    print(" قم بتدريب النموذج أولاً باستخدام train_model.py")
    exit()

recognizer.read(trainer_path)

# تحميل كاشف الوجوه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# تعريف الأسماء المرتبطة بمعرفاتهم
names = {
    0: "غير معروف",
    1: "أحمد",
    2: "محمد",
    # أضف المزيد حسب الحاجة
}

# محاولة فتح الكاميرا
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("خطأ: تعذر الوصول إلى الكاميرا!")
    exit()

# ضبط دقة الكاميرا لتحسين الأداء
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX
last_recognition_time = time.time()

print("\n [INFO] جاري بدء التعرف على الوجوه...")
print("اضغط على زر ESC للخروج")

while True:
    # قراءة إطار من الكاميرا
    ret, frame = cap.read()
    if not ret:
        print("خطأ: تعذر قراءة إطار من الكاميرا!")
        break
    
    # تحويل الإطار إلى تدرج الرمادي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين للإضاءة المنخفضة
    gray = cv2.equalizeHist(gray)
    
    # كشف الوجوه
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    current_time = time.time()
    
    for (x, y, w, h) in faces:
        # التعرف على الوجه كل 0.5 ثانية فقط لتحسين الأداء
        if current_time - last_recognition_time > 0.5:
            # استخراج منطقة الوجه
            face_roi = gray[y:y+h, x:x+w]
            
            # تغيير حجم الصورة لتتناسب مع النموذج
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # التعرف على الوجه
            id, confidence = recognizer.predict(face_roi)
            last_recognition_time = current_time
            
            # التحقق من الثقة
            if confidence < 70:
                name = names.get(id, "غير معروف")
                confidence_text = f"الدقة: {round(100 - confidence)}%"
            else:
                name = "غير معروف"
                confidence_text = ""
        
        # رسم مستطيل حول الوجه
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # عرض اسم المستخدم
        cv2.putText(frame, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
        
        # عرض نسبة الدقة (إذا كانت معروفة)
        if confidence_text:
            cv2.putText(frame, confidence_text, (x+5, y+h-5), font, 0.5, (255, 255, 0), 1)
    
    # عرض عدد الوجوه المكتشفة
    cv2.putText(frame, f"الوجوه: {len(faces)}", (10, 30), font, 0.7, (0, 0, 255), 2)
    
    # عرض النتيجة
    cv2.imshow('التعرف على الوجه - كاميرا اللابتوب', frame)
    
    # الخروج بالضغط على زر ESC
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# تحرير الموارد
cap.release()
cv2.destroyAllWindows()
print("\n [INFO] تم إيقاف النظام")
