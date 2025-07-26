# مشروع التعرف على الوجوه باستخدام OpenCV

هذا المشروع يطبق نظاماً بسيطاً للتعرف على الوجوه باستخدام مكتبة OpenCV.

## المتطلبات
- Python 3.8+
- OpenCV: `pip install opencv-python`
- NumPy: `pip install numpy`
- Pillow: `pip install pillow`

خطوات التشغيل
جمع بيانات الوجه:


python create_dataset.py
تدريب النموذج:


python train_model.py
تشغيل نظام التعرف:

bash
python recognizer.py
استكشاف الأخطاء الإضافية
تثبيت إصدار أقدم من OpenCV:


pip uninstall opencv-contrib-python -y
pip install opencv-contrib-python==4.5.5.64
استخدام بيئة افتراضية جديدة:


python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
تحديث حزمة setuptools:

bash
pip install --upgrade setuptools
إعادة تشغيل جهازك بعد التثبيت.

هذه التعديلات تضمن أن المشروع سيعمل بشكل صحيح على كاميرا اللابتوب 

