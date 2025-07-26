import os
import cv2
import numpy as np
from PIL import Image
import time

# مسار مجلد البيانات
dataset_path = "dataset"

# تهيئة كاشف الوجه ومتعرف الوجه
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    face_samples = []
    ids = []
    
    print("\n [INFO] معالجة صور التدريب...")
    
    for image_path in image_paths:
        # تحميل الصورة وتحويلها إلى تدرج الرمادي
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        
        # استخراج معرف المستخدم من اسم الملف
        try:
            user_id = int(os.path.split(image_path)[-1].split(".")[1])
        except:
            continue
        
        # تحسين جودة الصورة
        img_np = cv2.equalizeHist(img_np)
        
        # كشف الوجوه في الصورة
        faces = face_cascade.detectMultiScale(img_np)
        
        for (x, y, w, h) in faces:
            face_roi = img_np[y:y+h, x:x+w]
            
            # تغيير حجم الصورة لتوحيد الحجم
            face_roi = cv2.resize(face_roi, (200, 200))
            
            face_samples.append(face_roi)
            ids.append(user_id)
    
    return face_samples, np.array(ids)

# الحصول على بيانات التدريب
faces, ids = get_images_and_labels(dataset_path)

if len(faces) == 0:
    print("\n [ERROR] لم يتم العثور على أي صور للتدريب!")
    print(" تأكد من جمع الصور أولاً باستخدام create_dataset.py")
    exit()

# تدريب المتعرف
print("\n [INFO] تدريب النموذج...")
start_time = time.time()
recognizer.train(faces, ids)
training_time = time.time() - start_time

# حفظ النموذج المدرب
trainer_path = "trainer"
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer.write(os.path.join(trainer_path, "trainer.yml"))

# طباعة ملخص التدريب
print(f"\n [INFO] تم تدريب النموذج على {len(np.unique(ids))} شخص(أشخاص)")
print(f" [INFO] عدد الصور المستخدمة: {len(faces)}")
print(f" [INFO] زمن التدريب: {training_time:.2f} ثانية")
print(" [INFO] تم حفظ النموذج في trainer/trainer.yml")
