import os
import json
import uuid
from datetime import datetime

from flask import Flask, render_template, redirect, url_for, flash, request, send_file, abort
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from werkzeug.utils import secure_filename

from config import Config, BASE_DIR
from extensions import db, login_manager
from models import User, Prediction
from forms import RegisterForm, LoginForm, UploadForm

from ml.demo_support_set import RARE_DISEASE_CLASSES, ensure_support_set
from ml.prototypical import classify_query, build_prototypes
from ml.gradcam import generate_gradcam, overlay_heatmap_on_image


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    for folder in [
        app.config["UPLOAD_FOLDER"],
        app.config["HEATMAP_FOLDER"],
        app.config["SUPPORT_SET_FOLDER"],
        app.config["REPORT_FOLDER"],
        os.path.join(BASE_DIR, "instance"),
    ]:
        os.makedirs(folder, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    with app.app_context():
        db.create_all()
        ensure_support_set(app.config["SUPPORT_SET_FOLDER"], k_shot=app.config["K_SHOT"])
        # warm the prototype cache at startup
        build_prototypes(app.config["SUPPORT_SET_FOLDER"], app.config["EMBEDDING_DIM"], app.config["K_SHOT"])

    register_routes(app)
    return app


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def register_routes(app):

    @app.route("/")
    def index():
        return render_template("index.html", classes=RARE_DISEASE_CLASSES)

    # ---------------- AUTH ----------------

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        form = RegisterForm()
        if form.validate_on_submit():
            existing = User.query.filter_by(email=form.email.data.lower().strip()).first()
            if existing:
                flash("An account with that email already exists. Please log in.", "warning")
                return redirect(url_for("login"))
            user = User(full_name=form.full_name.data.strip(), email=form.email.data.lower().strip())
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash("Account created successfully! You can now log in.", "success")
            return redirect(url_for("login"))
        return render_template("register.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data.lower().strip()).first()
            if user and user.check_password(form.password.data):
                login_user(user)
                flash(f"Welcome back, {user.full_name}!", "success")
                next_page = request.args.get("next")
                return redirect(next_page or url_for("dashboard"))
            flash("Invalid email or password.", "danger")
        return render_template("login.html", form=form)

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out.", "info")
        return redirect(url_for("index"))

    # ---------------- APP ----------------

    @app.route("/dashboard")
    @login_required
    def dashboard():
        recent = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.desc())
            .limit(6)
            .all()
        )
        total = Prediction.query.filter_by(user_id=current_user.id).count()
        class_counts = {}
        for p in Prediction.query.filter_by(user_id=current_user.id).all():
            class_counts[p.predicted_class] = class_counts.get(p.predicted_class, 0) + 1
        return render_template(
            "dashboard.html", recent=recent, total=total, class_counts=class_counts, classes=RARE_DISEASE_CLASSES
        )

    @app.route("/support-set")
    @login_required
    def support_set_view():
        support_root = app.config["SUPPORT_SET_FOLDER"]
        data = {}
        for cls in RARE_DISEASE_CLASSES:
            class_dir = os.path.join(support_root, cls.replace(" ", "_"))
            files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg")))[:5]
            data[cls] = [f"support_set/{cls.replace(' ', '_')}/{f}" for f in files]
        return render_template("support_set.html", data=data, k_shot=app.config["K_SHOT"])

    @app.route("/upload", methods=["GET", "POST"])
    @login_required
    def upload():
        form = UploadForm()
        if form.validate_on_submit():
            file = form.image.data
            if not allowed_file(file.filename):
                flash("Unsupported file type.", "danger")
                return redirect(url_for("upload"))

            unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(save_path)

            # ---- Few-shot inference (Prototypical Network) ----
            result = classify_query(
                save_path,
                app.config["SUPPORT_SET_FOLDER"],
                embedding_dim=app.config["EMBEDDING_DIM"],
                k_shot=app.config["K_SHOT"],
            )
            prototypes, _ = build_prototypes(
                app.config["SUPPORT_SET_FOLDER"], app.config["EMBEDDING_DIM"], app.config["K_SHOT"]
            )
            predicted_prototype = prototypes[result["predicted_class"]]

            # ---- Grad-CAM explanation ----
            cam = generate_gradcam(
                result["query_tensor"], predicted_prototype, embedding_dim=app.config["EMBEDDING_DIM"]
            )
            heatmap_img = overlay_heatmap_on_image(result["pil_image"], cam)
            heatmap_name = f"heatmap_{unique_name.rsplit('.', 1)[0]}.png"
            heatmap_img.save(os.path.join(app.config["HEATMAP_FOLDER"], heatmap_name))

            prediction = Prediction(
                user_id=current_user.id,
                patient_ref=form.patient_ref.data.strip() if form.patient_ref.data else None,
                original_image=unique_name,
                heatmap_image=heatmap_name,
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                class_distances_json=json.dumps(result["distances"]),
                class_probs_json=json.dumps(result["probabilities"]),
                n_way=len(RARE_DISEASE_CLASSES),
                k_shot=app.config["K_SHOT"],
                notes=form.notes.data,
            )
            db.session.add(prediction)
            db.session.commit()

            flash("Analysis complete!", "success")
            return redirect(url_for("result", prediction_id=prediction.id))

        return render_template("upload.html", form=form, classes=RARE_DISEASE_CLASSES)

    @app.route("/result/<int:prediction_id>")
    @login_required
    def result(prediction_id):
        prediction = Prediction.query.get_or_404(prediction_id)
        if prediction.user_id != current_user.id:
            abort(403)
        probs = json.loads(prediction.class_probs_json)
        distances = json.loads(prediction.class_distances_json)
        sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        return render_template(
            "result.html",
            prediction=prediction,
            sorted_probs=sorted_probs,
            distances=distances,
        )

    @app.route("/history")
    @login_required
    def history():
        predictions = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.desc())
            .all()
        )
        return render_template("history.html", predictions=predictions)

    @app.route("/prediction/<int:prediction_id>/delete", methods=["POST"])
    @login_required
    def delete_prediction(prediction_id):
        prediction = Prediction.query.get_or_404(prediction_id)
        if prediction.user_id != current_user.id:
            abort(403)
        db.session.delete(prediction)
        db.session.commit()
        flash("Record deleted.", "info")
        return redirect(url_for("history"))

    @app.route("/report/<int:prediction_id>")
    @login_required
    def download_report(prediction_id):
        prediction = Prediction.query.get_or_404(prediction_id)
        if prediction.user_id != current_user.id:
            abort(403)
        from report import build_pdf_report
        pdf_path = build_pdf_report(app, prediction, current_user)
        return send_file(pdf_path, as_attachment=True, download_name=f"report_{prediction.id}.pdf")

    @app.errorhandler(403)
    def forbidden(e):
        return render_template("error.html", code=403, message="You don't have access to that record."), 403

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.context_processor
    def inject_now():
        return {"current_year": datetime.utcnow().year}


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5014)