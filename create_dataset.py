import cv2
import os

# إنشاء مجلد dataset إذا لم يكن موجوداً
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# إدخال معرف المستخدم
user_id = input("أدخل المعرف الرقمي للشخص (مثال: 1, 2, 3): ")
user_name = input("أدخل اسم الشخص: ")
count = 0

# تحميل مصنف الوجه
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# بدء تشغيل الكاميرا
cap = cv2.VideoCapture(0)

print("\n [INFO] جاري بدء جمع صور الوجه...")
print("اضغط على زر ESC لإيقاف الجمع")

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
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        # رسم مستطيل حول الوجه
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # زيادة العداد
        count += 1
        
        # حفظ صورة الوجه
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{dataset_path}/User.{user_id}.{count}.jpg", face_img)
        
        # عرض العداد على الشاشة
        cv2.putText(frame, str(count), (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # عرض الإطار
    cv2.imshow('جمع بيانات الوجه', frame)
    
    # الخروج بالضغط على زر ESC أو عند جمع 100 صورة
    k = cv2.waitKey(100) & 0xff
    if k == 27 or count >= 100:
        break

# تحرير الموارد
cap.release()
cv2.destroyAllWindows()
print(f"\n [INFO] تم جمع {count} صورة لوجه {user_name} في مجلد dataset")