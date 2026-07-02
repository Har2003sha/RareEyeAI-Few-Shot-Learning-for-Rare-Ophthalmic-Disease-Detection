from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="clinician")  # clinician / researcher / admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship(
        "Prediction", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    patient_ref = db.Column(db.String(120), nullable=True)
    original_image = db.Column(db.String(255), nullable=False)
    heatmap_image = db.Column(db.String(255), nullable=True)

    predicted_class = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    class_distances_json = db.Column(db.Text, nullable=True)  # JSON string of {class: distance}
    class_probs_json = db.Column(db.Text, nullable=True)      # JSON string of {class: prob}

    n_way = db.Column(db.Integer, default=5)
    k_shot = db.Column(db.Integer, default=5)

    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)