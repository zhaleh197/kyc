# مرحله 1: انتخاب تصویر پایه سبک
FROM python:3.12-slim

# مرحله 2: تنظیم مسیر کاری داخل کانتینر
WORKDIR /app

# مرحله 3: کپی کردن فایل‌های پروژه به کانتینر
COPY . /app

# مرحله 4: نصب پیش‌نیازها
RUN pip install --no-cache-dir -r requirements.txt

# مرحله 5: تعریف پورت
EXPOSE 5000

# مرحله 6: اجرای برنامه
CMD ["python", "phase1F0.py"]
