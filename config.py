
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "rare-eye-disease-few-shot-secret-key-change-me")

    # By default we use SQLite so the app runs out of the box.
    # To use PostgreSQL instead, set the DATABASE_URL environment variable, e.g.:
    #   postgresql://username:password@localhost:5432/ophthalmic_db
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'app.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")
    SUPPORT_SET_FOLDER = os.path.join(BASE_DIR, "static", "support_set")
    REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

    # Few-shot learning settings
    N_WAY = 5          # number of rare disease classes in the demo episode
    K_SHOT = 5          # number of support images per class
    EMBEDDING_DIM = 512  # ResNet-18 pooled feature dimension