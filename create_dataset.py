import cv2
import os
import time

# إنشاء مجلد dataset إذا لم يكن موجوداً
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# إدخال بيانات المستخدم
user_id = input("أدخل المعرف الرقمي للشخص (مثال: 1, 2, 3): ")
user_name = input("أدخل اسم الشخص: ")
count = 0
max_images = 100  # الحد الأقصى لعدد الصور

# تحميل مصنف الوجه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# محاولة فتح الكاميرا
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("خطأ: تعذر الوصول إلى الكاميرا!")
    exit()

# ضبط دقة الكاميرا لتحسين الأداء
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n [INFO] جاري بدء جمع صور الوجه...")
print("اضغط على زر ESC لإيقاف الجمع")
print("توجيه الوجه نحو الكاميرا مع إضاءة جيدة")

start_time = time.time()
last_capture_time = start_time

while count < max_images:
    # قراءة إطار من الكاميرا
    ret, frame = cap.read()
    if not ret:
        print("خطأ: تعذر قراءة إطار من الكاميرا!")
        break
    
    # تحويل الإطار إلى تدرج الرمادي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين للإضاءة المنخفضة
    gray = cv2.equalizeHist(gray)
    
    # كشف الوجوه مع معلمات محسنة
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        # رسم مستطيل حول الوجه
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # التقاط صورة كل 0.3 ثانية
        current_time = time.time()
        if current_time - last_capture_time > 0.3:
            count += 1
            last_capture_time = current_time
            
            # حفظ صورة الوجه
            face_img = gray[y:y+h, x:x+w]
            
            # تغيير حجم الصورة لتوحيد الحجم
            face_img = cv2.resize(face_img, (200, 200))
            
            # حفظ الصورة
            cv2.imwrite(f"{dataset_path}/User.{user_id}.{count}.jpg", face_img)
        
        # عرض العداد على الشاشة
        cv2.putText(frame, f"الصور: {count}/{max_images}", (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # عرض الوقت المنقضي
    elapsed_time = int(time.time() - start_time)
    cv2.putText(frame, f"الوقت: {elapsed_time} ثانية", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # عرض الإطار
    cv2.imshow('جمع بيانات الوجه', frame)
    
    # الخروج بالضغط على زر ESC
    if cv2.waitKey(1) == 27:
        break

# تحرير الموارد
cap.release()
cv2.destroyAllWindows()
print(f"\n [INFO] تم جمع {count} صورة لوجه {user_name}")
