# مشروع التعرف على الوجوه باستخدام OpenCV

هذا المشروع يطبق نظاماً بسيطاً للتعرف على الوجوه باستخدام مكتبة OpenCV.

## المتطلبات
- Python 3.8+
- OpenCV: `pip install opencv-python`
- NumPy: `pip install numpy`
- Pillow: `pip install pillow`

خطوات التشغيل
جمع بيانات الوجوه: python create_dataset.py

تدريب النموذج: python train_model.py

التعرف على الوجوه: python recognizer.py

نصائح للاستخدام
أثناء جمع البيانات:

تأكد من إضاءة جيدة

غير زوايا وجهك لتحسين الدقة

اجمع 100 صورة على الأقل لكل شخص

لتحسين الدقة:

أضف المزيد من الأشخاص بتشغيل create_dataset.py بأرقام معرفات جديدة

عدّل ملف recognizer.py وأضف الأسماء في قاموس names

في حال مشاكل الكاميرا:

جرب تغيير VideoCapture(0) إلى VideoCapture(1)

تأكد من عدم استخدام الكاميرا من برامج أخرى
