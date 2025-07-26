import cv2
import numpy as np
import os

# تحميل النموذج المدرب
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# تحميل كاشف الوجوه
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# تعريف الأسماء المرتبطة بمعرفاتهم
names = {
    0: "غير معروف",
    1: "أحمد",
    2: "محمد",
    # أضف المزيد حسب الحاجة
}

# بدء تشغيل الكاميرا
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

print("\n [INFO] جاري بدء التعرف على الوجوه...")
print("اضغط على زر ESC للخروج")

while True:
    # قراءة إطار من الكاميرا
    ret, frame = cap.read()
    if not ret:
        break
    
    # تحويل الإطار إلى تدرج الرمادي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # كشف الوجوه
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )
    
    for (x, y, w, h) in faces:
        # التعرف على الوجه
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
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
    
    # عرض النتيجة
    cv2.imshow('التعرف على الوجه', frame)
    
    # الخروج بالضغط على زر ESC
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# تحرير الموارد
cap.release()
cv2.destroyAllWindows()
print("\n [INFO] تم إيقاف النظام")