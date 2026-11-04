"""
ReliAttend AI — Database Seeding Script
Creates default Administrator and Faculty test accounts with secure bcrypt hashes.
Run: python -m scripts.seed_users
"""

import os
import sys

# Ensure backend root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import SessionLocal, init_db
from app.core.security import hash_password
from app.models.user import User


def seed_default_users() -> None:
    """Seed initial default Administrator and Faculty users if they do not exist."""
    init_db()
    db = SessionLocal()
    try:
        # 1. Administrator Account
        admin = db.query(User).filter_by(username="admin").first()
        if not admin:
            admin = User(
                email="admin@reliattend.com",
                username="admin",
                password_hash=hash_password("AdminPassword@123"),
                full_name="System Administrator",
                role="admin",
                is_active=True,
            )
            db.add(admin)
            print("  Created default Admin: admin@reliattend.com (username: admin, pass: AdminPassword@123)")
        else:
            print("  Admin user already exists.")

        # 2. Faculty Account
        faculty = db.query(User).filter_by(username="dr_sharma").first()
        if not faculty:
            faculty = User(
                email="faculty@reliattend.com",
                username="dr_sharma",
                password_hash=hash_password("FacultyPassword@123"),
                full_name="Dr. Ramesh Sharma",
                role="faculty",
                is_active=True,
            )
            db.add(faculty)
            print("  Created default Faculty: faculty@reliattend.com (username: dr_sharma, pass: FacultyPassword@123)")
        else:
            print("  Faculty user already exists.")

        db.commit()
        print("Database seeding completed successfully.")
    except Exception as exc:
        db.rollback()
        print(f"Error seeding database: {exc}")
        raise exc
    finally:
        db.close()


if __name__ == "__main__":
    seed_default_users()

