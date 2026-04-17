from pathlib import Path
from io import BytesIO
import pymysql
import joblib

BASE_DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "charset": "utf8mb4",
    "connect_timeout": 60,
    "read_timeout": 600,
    "write_timeout": 600,
    "autocommit": True
}

DB_NAME = "thunder_storm_db"
TABLE_NAME = "ml_models"
MODEL_DIR = Path("/Users/mast/Documents/VInayPrograming/Thunder_storm_Predi/models")


def get_server_connection():
    return pymysql.connect(
        host=BASE_DB_CONFIG["host"],
        user=BASE_DB_CONFIG["user"],
        password=BASE_DB_CONFIG["password"],
        charset=BASE_DB_CONFIG["charset"],
        connect_timeout=BASE_DB_CONFIG["connect_timeout"],
        read_timeout=BASE_DB_CONFIG["read_timeout"],
        write_timeout=BASE_DB_CONFIG["write_timeout"],
        autocommit=BASE_DB_CONFIG["autocommit"]
    )


def get_db_connection():
    return pymysql.connect(
        host=BASE_DB_CONFIG["host"],
        user=BASE_DB_CONFIG["user"],
        password=BASE_DB_CONFIG["password"],
        database=DB_NAME,
        charset=BASE_DB_CONFIG["charset"],
        connect_timeout=BASE_DB_CONFIG["connect_timeout"],
        read_timeout=BASE_DB_CONFIG["read_timeout"],
        write_timeout=BASE_DB_CONFIG["write_timeout"],
        autocommit=BASE_DB_CONFIG["autocommit"]
    )


def create_database():
    conn = get_server_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        conn.close()


def create_table():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL UNIQUE,
                    file_extension VARCHAR(20) NOT NULL,
                    model_data LONGBLOB NOT NULL,
                    file_size BIGINT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                )
            """)
    finally:
        conn.close()


def setup_database():
    create_database()
    create_table()


def save_one_model(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"[SKIP] File not found: {file_path}")
        return False

    with open(file_path, "rb") as f:
        binary_data = f.read()

    file_size = len(binary_data)
    print(f"[INFO] Trying: {file_path.name} ({file_size / 1024:.2f} KB)")

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            query = f"""
                INSERT INTO `{TABLE_NAME}` (
                    model_name,
                    file_extension,
                    model_data,
                    file_size
                )
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    file_extension = VALUES(file_extension),
                    model_data = VALUES(model_data),
                    file_size = VALUES(file_size),
                    created_at = CURRENT_TIMESTAMP
            """
            cursor.execute(
                query,
                (
                    file_path.name,
                    file_path.suffix,
                    binary_data,
                    file_size
                )
            )
        print(f"[OK] Saved: {file_path.name}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed: {file_path.name} -> {e}")
        return False

    finally:
        if conn:
            conn.close()


def save_all_models():
    if not MODEL_DIR.exists():
        print(f"[ERROR] Model directory not found: {MODEL_DIR}")
        return

    files = sorted(MODEL_DIR.glob("*.joblib"))

    if not files:
        print("[ERROR] No .joblib files found in the models folder.")
        return

    success = 0
    failed = 0

    for file in files:
        if save_one_model(file):
            success += 1
        else:
            failed += 1

    print(f"\nFinished | Success: {success} | Failed: {failed}")


def list_models():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT id, model_name, file_extension, file_size, created_at
                FROM `{TABLE_NAME}`
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()

        if not rows:
            print("\nNo models found in database.")
            return

        print("\nModels in database:")
        for row in rows:
            print(
                f"ID: {row[0]} | "
                f"Name: {row[1]} | "
                f"Ext: {row[2]} | "
                f"Size: {row[3]} bytes | "
                f"Created: {row[4]}"
            )
    finally:
        conn.close()


def load_model_from_db(model_name):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT model_data FROM `{TABLE_NAME}` WHERE model_name = %s",
                (model_name,)
            )
            row = cursor.fetchone()

        if not row:
            raise ValueError(f"Model '{model_name}' not found in database.")

        model_bytes = row[0]
        return joblib.load(BytesIO(model_bytes))
    finally:
        conn.close()


if __name__ == "__main__":
    setup_database()
    save_all_models()
    list_models()