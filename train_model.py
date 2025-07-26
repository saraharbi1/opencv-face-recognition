import os
import cv2
import numpy as np
from PIL import Image

# مسار مجلد البيانات
dataset_path = "dataset"

# تهيئة كاشف الوجه ومتعرف الوجه
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        
        # كشف الوجوه في الصورة
        faces = face_cascade.detectMultiScale(img_np)
        
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(user_id)
    
    return face_samples, np.array(ids)

# الحصول على بيانات التدريب
faces, ids = get_images_and_labels(dataset_path)

# تدريب المتعرف
print("\n [INFO] تدريب النموذج... قد يستغرق بضع ثوانٍ")
recognizer.train(faces, ids)

# حفظ النموذج المدرب
trainer_path = "trainer"
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer.write(os.path.join(trainer_path, "trainer.yml"))

# طباعة ملخص التدريب
print(f"\n [INFO] تم تدريب النموذج على {len(np.unique(ids))} شخص(أشخاص)")
print(" [INFO] تم حفظ النموذج في trainer/trainer.yml")