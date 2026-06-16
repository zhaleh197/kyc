from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 🟢 اطلاعات اتصال به PostgreSQL
DATABASE_URL = "postgresql://postgres:s123456h@localhost/face_db"

# 🔧 ساخت engine
engine = create_engine(DATABASE_URL)

# 🧵 ساخت SessionLocal برای استفاده در CRUD
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ⚙️ تعریف base برای مدل‌های ORM
Base = declarative_base()
# تابعی برای گرفتن session و بستن آن در پایان
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()