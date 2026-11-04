"""
ReliAttend AI — Safe SQLite Multi-Reference Migration Script
Converts face_references table from 1-to-1 unique student_id to 1-to-many.
Adds pose_label and is_primary columns.
Guarantees zero data loss: backups existing database before applying changes.
"""

import os
import shutil
import sqlite3
from datetime import datetime


def migrate_database(db_path: str = "reliattend.db") -> bool:
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' does not exist. Nothing to migrate.")
        return True

    # 1. Create a safe backup before migration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    shutil.copy2(db_path, backup_path)
    print(f"Created pre-migration backup at: {backup_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        # Check current columns in face_references
        cur.execute("PRAGMA table_info(face_references)")
        columns = [row[1] for row in cur.fetchall()]
        print(f"Existing columns in face_references: {columns}")

        has_pose_label = "pose_label" in columns
        has_is_primary = "is_primary" in columns

        # Check if student_id is unique
        cur.execute("PRAGMA index_list(face_references)")
        indices = cur.fetchall()
        print(f"Existing indices: {indices}")

        # Start transaction
        cur.execute("BEGIN TRANSACTION;")

        # Create temporary table without unique constraint on student_id
        cur.execute("""
            CREATE TABLE face_references_multiref (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                image_path VARCHAR(500) NOT NULL,
                embedding BLOB NOT NULL,
                embedding_size INTEGER DEFAULT 128 NOT NULL,
                model_name VARCHAR(100) DEFAULT 'opencv_sface' NOT NULL,
                model_version VARCHAR(50) DEFAULT '2021dec' NOT NULL,
                quality_score FLOAT,
                pose_label VARCHAR(50) DEFAULT 'frontal' NOT NULL,
                is_primary BOOLEAN DEFAULT 0 NOT NULL,
                status VARCHAR(20) DEFAULT 'active' NOT NULL,
                enrolled_at DATETIME NOT NULL,
                enrolled_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
                deactivated_at DATETIME
            );
        """)

        # Copy existing data
        if has_pose_label and has_is_primary:
            cur.execute("""
                INSERT INTO face_references_multiref (
                    id, student_id, image_path, embedding, embedding_size,
                    model_name, model_version, quality_score, pose_label,
                    is_primary, status, enrolled_at, enrolled_by, deactivated_at
                )
                SELECT
                    id, student_id, image_path, embedding, embedding_size,
                    model_name, model_version, quality_score, pose_label,
                    is_primary, status, enrolled_at, enrolled_by, deactivated_at
                FROM face_references;
            """)
        else:
            cur.execute("""
                INSERT INTO face_references_multiref (
                    id, student_id, image_path, embedding, embedding_size,
                    model_name, model_version, quality_score, pose_label,
                    is_primary, status, enrolled_at, enrolled_by, deactivated_at
                )
                SELECT
                    id, student_id, image_path, embedding, embedding_size,
                    model_name, model_version, quality_score, 'frontal',
                    1, status, enrolled_at, enrolled_by, deactivated_at
                FROM face_references;
            """)

        # Drop old table
        cur.execute("DROP TABLE face_references;")

        # Rename new table
        cur.execute("ALTER TABLE face_references_multiref RENAME TO face_references;")

        # Create standard index on student_id (non-unique)
        cur.execute("CREATE INDEX ix_face_references_student_id ON face_references (student_id);")
        cur.execute("CREATE INDEX ix_face_references_id ON face_references (id);")

        conn.commit()
        print("Migration applied successfully! face_references is now 1-to-many.")
        return True

    except Exception as exc:
        conn.rollback()
        print(f"Migration error: {exc}. Rolled back.")
        # Restore backup
        shutil.copy2(backup_path, db_path)
        print("Restored original database from backup.")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    migrate_database()

