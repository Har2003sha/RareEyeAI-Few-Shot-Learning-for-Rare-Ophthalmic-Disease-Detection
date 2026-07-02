from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Length, EqualTo


class RegisterForm(FlaskForm):
    full_name = StringField("Full Name", validators=[DataRequired(), Length(min=2, max=120)])
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField(
        "Confirm Password", validators=[DataRequired(), EqualTo("password", message="Passwords must match")]
    )
    submit = SubmitField("Create Account")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Log In")


class UploadForm(FlaskForm):
    patient_ref = StringField("Patient / Case Reference (optional)", validators=[Length(max=120)])
    image = FileField(
        "Retinal Fundus Image",
        validators=[FileRequired(), FileAllowed(["jpg", "jpeg", "png", "bmp", "tif", "tiff"], "Images only!")],
    )
    notes = TextAreaField("Clinical Notes (optional)")
    submit = SubmitField("Run Few-Shot Detection")