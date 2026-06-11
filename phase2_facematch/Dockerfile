FROM python:3.10-slim

# مرحله 2: تنظیم مسیر کاری داخل کانتینر
WORKDIR /app

# مرحله 3: کپی کردن فایل‌های پروژه به کانتینر
COPY . /app

# مرحله 4: نصب پیش‌نیازها
RUN pip install --no-cache-dir -r requirements.txt

# مرحله 5: تعریف پورت
EXPOSE 9000


CMD ["uvicorn", "facematching:app", "--host", "0.0.0.0", "--port", "9000"]
