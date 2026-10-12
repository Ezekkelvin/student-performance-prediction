
import os
import calendar
import html
from datetime import date, datetime, time, timedelta
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
import resend
from dotenv import load_dotenv
from flask import (
    Flask, flash, redirect, render_template, request, url_for, abort
)
from flask_login import (
    LoginManager, UserMixin, current_user, login_user,
    logout_user, login_required
)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash


# ---------------------------------------------------------
# APPLICATION CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-only-change-this-secret")

# Prefer PostgreSQL in production, while retaining SQLite locally.
database_url = (
    os.getenv("DATABASE_URL")
    or os.getenv("POSTGRES_URL")
    or ""
).strip()

# Use the installed psycopg 3 driver for PostgreSQL URLs.
if database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://", "postgresql+psycopg://", 1
    )
elif database_url.startswith("postgresql://"):
    database_url = database_url.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

if database_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
elif os.getenv("VERCEL"):
    raise RuntimeError(
        "PostgreSQL is not configured. Set DATABASE_URL or POSTGRES_URL in Vercel."
    )
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"sqlite:///{BASE_DIR / 'student_performance.db'}"
    )

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    os.getenv("COOKIE_SECURE", "0") == "1"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    os.getenv("COOKIE_SECURE", "0") == "1"
)

db = SQLAlchemy(app)
csrf = CSRFProtect(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please sign in to continue."
login_manager.login_message_category = "info"


# ---------------------------------------------------------
# RESEND EMAIL CONFIGURATION
# ---------------------------------------------------------

resend.api_key = (os.getenv("RESEND_API_KEY") or "").strip()
MAIL_FROM = (os.getenv("MAIL_FROM") or "").strip()
RESEND_TEST_RECIPIENT = (
    os.getenv("RESEND_TEST_RECIPIENT") or ""
).strip()


# ---------------------------------------------------------
# PROJECT SETTINGS
# ---------------------------------------------------------

PARTICIPATION_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

LEVELS = [
    "100 Level",
    "200 Level",
    "300 Level",
    "400 Level",
    "500 Level"
]

DEFAULT_COURSES = [
    "Software Engineering",
    "Database Systems",
    "Computer Networks",
    "Web Application Development"
]

MODEL_PATH = BASE_DIR / "student_performance_model.pkl"

try:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
except Exception:
    app.logger.exception("Could not load the student performance model.")
    model = None


# ---------------------------------------------------------
# DATABASE MODELS
# ---------------------------------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(
        db.String(255), unique=True, nullable=False, index=True
    )
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    status = db.Column(
        db.String(20), nullable=False, default="approved"
    )
    student_id = db.Column(
        db.String(60), unique=True, nullable=True, index=True
    )
    level = db.Column(db.String(20), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    lecturer_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=True
    )
    lecturer = db.relationship("User", foreign_keys=[lecturer_id])


class Enrollment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    course_id = db.Column(
        db.Integer, db.ForeignKey("course.id"), nullable=False
    )
    enrolled_at = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False
    )

    student = db.relationship("User", backref="enrollments")
    course = db.relationship("Course", backref="enrollments")

    __table_args__ = (
        db.UniqueConstraint(
            "student_id", "course_id", name="uq_student_course"
        ),
    )


class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lecturer_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    course_id = db.Column(
        db.Integer, db.ForeignKey("course.id"), nullable=False
    )
    weekday = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    first_date = db.Column(db.Date, nullable=False)
    last_date = db.Column(db.Date, nullable=False)
    room = db.Column(db.String(100), nullable=True)

    lecturer = db.relationship("User")
    course = db.relationship("Course")


class ClassSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    schedule_id = db.Column(
        db.Integer, db.ForeignKey("schedule.id"), nullable=False
    )
    course_id = db.Column(
        db.Integer, db.ForeignKey("course.id"), nullable=False
    )
    lecturer_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    session_date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    room = db.Column(db.String(100), nullable=True)
    status = db.Column(
        db.String(20), nullable=False, default="scheduled"
    )
    opened_at = db.Column(db.DateTime, nullable=True)
    closed_at = db.Column(db.DateTime, nullable=True)

    schedule = db.relationship("Schedule", backref="sessions")
    course = db.relationship("Course")
    lecturer = db.relationship("User")

    __table_args__ = (
        db.UniqueConstraint(
            "schedule_id",
            "session_date",
            name="uq_schedule_session_date"
        ),
    )


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("class_session.id"), nullable=False
    )
    student_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    status = db.Column(
        db.String(20), nullable=False, default="pending"
    )
    marked_at = db.Column(db.DateTime, nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    reviewed_by_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=True
    )

    session = db.relationship(
        "ClassSession", backref="attendance_records"
    )
    student = db.relationship("User", foreign_keys=[student_id])
    reviewer = db.relationship("User", foreign_keys=[reviewed_by_id])

    __table_args__ = (
        db.UniqueConstraint(
            "session_id",
            "student_id",
            name="uq_attendance_session_student"
        ),
    )


class AcademicRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    course_id = db.Column(
        db.Integer, db.ForeignKey("course.id"), nullable=False
    )
    assignment = db.Column(db.Float, nullable=False, default=0)
    practical = db.Column(db.Float, nullable=False, default=0)
    ca = db.Column(db.Float, nullable=False, default=0)
    previous_cgpa = db.Column(db.Float, nullable=False, default=0)
    participation = db.Column(
        db.String(10), nullable=False, default="Medium"
    )
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    student = db.relationship("User")
    course = db.relationship("Course")

    __table_args__ = (
        db.UniqueConstraint(
            "student_id",
            "course_id",
            name="uq_academic_student_course"
        ),
    )


# ---------------------------------------------------------
# LOGIN, ROLES AND SHARED HELPERS
# ---------------------------------------------------------

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def role_required(*roles):
    def decorator(fn):
        @wraps(fn)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                abort(403)

            if (
                current_user.role != "admin"
                and current_user.status != "approved"
            ):
                logout_user()
                flash(
                    "Your account is not approved yet.",
                    "warning"
                )
                return redirect(url_for("login"))

            return fn(*args, **kwargs)

        return wrapped

    return decorator


def dashboard_for(user):
    endpoints = {
        "student": "student_dashboard",
        "lecturer": "lecturer_dashboard",
        "advisor": "advisor_dashboard",
        "admin": "admin_dashboard"
    }
    return endpoints.get(user.role, "home")


def attendance_summary(student_id, course_id):
    """Count only closed sessions in the course attendance denominator."""
    total = ClassSession.query.filter_by(
        course_id=course_id,
        status="closed"
    ).count()

    present = (
        db.session.query(Attendance)
        .join(ClassSession)
        .filter(
            Attendance.student_id == student_id,
            ClassSession.course_id == course_id,
            ClassSession.status == "closed",
            Attendance.status == "present"
        )
        .count()
    )

    percentage = round(present / total * 100, 2) if total else 0.0
    return percentage, present, total


def ensure_academic_record(student_id, course_id):
    record = AcademicRecord.query.filter_by(
        student_id=student_id,
        course_id=course_id
    ).first()

    if record is None:
        record = AcademicRecord(
            student_id=student_id,
            course_id=course_id
        )
        db.session.add(record)
        db.session.commit()

    return record


def get_prediction(
    attendance, assignment, practical, ca, cgpa, participation
):
    if model is None:
        raise RuntimeError(
            "student_performance_model.pkl is missing from the project folder."
        )

    if participation not in PARTICIPATION_MAPPING:
        raise ValueError("Invalid class participation value.")

    if not 0 <= assignment <= 20:
        raise ValueError("Assignment score must be between 0 and 20.")

    if not 0 <= practical <= 20:
        raise ValueError("Practical score must be between 0 and 20.")

    if not 0 <= ca <= 20:
        raise ValueError("CA score must be between 0 and 20.")

    if not 0 <= cgpa <= 5:
        raise ValueError("Previous CGPA must be between 0 and 5.")

    # Keep these six features in the model's expected order.
    frame = pd.DataFrame(
        [[
            attendance,
            assignment,
            practical,
            ca,
            cgpa,
            PARTICIPATION_MAPPING[participation]
        ]],
        columns=[
            "Attendance",
            "Assignment_Score",
            "Practical_Score",
            "CA_Score",
            "Previous_CGPA",
            "Class_Participation"
        ]
    )

    prediction = str(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    return prediction, confidence


# ---------------------------------------------------------
# COMMAND-LINE ADMIN AND DATABASE COMMANDS
# ---------------------------------------------------------

@app.cli.command("init-db")
def init_db():
    """Create tables and seed the four default project courses."""
    db.create_all()

    for name in DEFAULT_COURSES:
        if not Course.query.filter_by(name=name).first():
            db.session.add(Course(name=name))

    db.session.commit()
    print("Database tables created and default courses seeded.")


@app.cli.command("create-admin")
def create_admin():
    """Interactively create an administrator; never public registration."""
    import getpass

    email = input("Admin email: ").strip().lower()
    full_name = input("Admin full name: ").strip()
    password = getpass.getpass(
        "Admin password (12+ characters recommended): "
    )

    if not email or not full_name or len(password) < 10:
        print(
            "Email, name, and a password of at least 10 characters are required."
        )
        return

    if User.query.filter_by(email=email).first():
        print("An account with that email already exists.")
        return

    db.create_all()

    admin = User(
        email=email,
        full_name=full_name,
        role="admin",
        status="approved"
    )
    admin.set_password(password)

    db.session.add(admin)
    db.session.commit()

    print(
        "Administrator created. Run 'flask --app app init-db' "
        "if courses are not seeded yet."
    )


@app.cli.command("check-admin")
def check_admin():
    """Check whether an administrator account exists."""
    email = input("Admin email to check: ").strip().lower()
    user = User.query.filter_by(email=email).first()

    if user is None:
        print(
            "No account found for that email in the database "
            "currently configured by this app."
        )
        return

    print(
        f"Account found: yes\n"
        f"Role: {user.role}\n"
        f"Status: {user.status}\n"
        f"Name: {user.full_name}"
    )


@app.cli.command("reset-admin-password")
def reset_admin_password():
    """Safely reset an existing administrator's password."""
    import getpass

    email = input("Admin email: ").strip().lower()
    user = User.query.filter_by(
        email=email,
        role="admin"
    ).first()

    if user is None:
        print(
            "No administrator account with that email exists "
            "in the currently configured database."
        )
        print(
            "No changes were made. Check DATABASE_URL and the email, "
            "then try again."
        )
        return

    password = getpass.getpass(
        "New admin password (at least 10 characters; input is hidden): "
    )
    confirmation = getpass.getpass("Confirm new admin password: ")

    if len(password) < 10:
        print("Password must be at least 10 characters. No changes made.")
        return

    if password != confirmation:
        print("Passwords did not match. No changes were made.")
        return

    user.set_password(password)
    db.session.commit()

    print("Administrator password updated successfully.")


# ---------------------------------------------------------
# PUBLIC HOMEPAGE AND ABOUT
# ---------------------------------------------------------

@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for(dashboard_for(current_user)))

    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ---------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    courses = Course.query.order_by(Course.name).all()

    if request.method == "POST":
        role = request.form.get("role", "student")
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if role not in ("student", "lecturer", "advisor"):
            flash("Choose a valid account type.", "danger")
            return render_template(
                "register.html", courses=courses, levels=LEVELS
            )

        if not full_name or not email or len(password) < 10:
            flash(
                "Enter your name and email, and use a password "
                "of at least 10 characters.",
                "danger"
            )
            return render_template(
                "register.html", courses=courses, levels=LEVELS
            )

        if User.query.filter_by(email=email).first():
            flash(
                "An account already exists with that email.",
                "danger"
            )
            return render_template(
                "register.html", courses=courses, levels=LEVELS
            )

        user = User(
            full_name=full_name,
            email=email,
            role=role,
            status="approved" if role == "student" else "pending"
        )
        user.set_password(password)

        course_ids = []

        if role == "student":
            sid = request.form.get("student_id", "").strip()
            level = request.form.get("level", "")
            course_ids = request.form.getlist("course_ids", type=int)

            if not sid or level not in LEVELS or not course_ids:
                flash(
                    "Students must provide a student ID, level, "
                    "and select at least one course.",
                    "danger"
                )
                return render_template(
                    "register.html", courses=courses, levels=LEVELS
                )

            if User.query.filter_by(student_id=sid).first():
                flash(
                    "That student ID is already registered. "
                    "Contact the administrator if this is an error.",
                    "danger"
                )
                return render_template(
                    "register.html", courses=courses, levels=LEVELS
                )

            valid_course_ids = {
                course.id for course in courses
            }
            if not set(course_ids).issubset(valid_course_ids):
                flash("One or more selected courses are invalid.", "danger")
                return render_template(
                    "register.html", courses=courses, levels=LEVELS
                )

            user.student_id = sid
            user.level = level

        db.session.add(user)

        try:
            db.session.flush()

            if role == "student":
                for cid in set(course_ids):
                    db.session.add(
                        Enrollment(
                            student_id=user.id,
                            course_id=cid
                        )
                    )

            db.session.commit()

        except IntegrityError:
            db.session.rollback()
            flash(
                "Could not create the account. Check the details and try again.",
                "danger"
            )
            return render_template(
                "register.html", courses=courses, levels=LEVELS
            )

        if role == "student":
            login_user(user)
            flash("Registration successful. Welcome!", "success")
            return redirect(url_for("student_dashboard"))

        flash(
            "Registration submitted. You can sign in after an "
            "administrator approves your account.",
            "success"
        )
        return redirect(url_for("login"))

    return render_template(
        "register.html", courses=courses, levels=LEVELS
    )


# ---------------------------------------------------------
# LOGIN AND LOGOUT
# ---------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for(dashboard_for(current_user)))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(
            request.form.get("password", "")
        ):
            flash("Invalid email or password.", "danger")

        elif user.status != "approved":
            flash(
                "Your account is pending approval or has been rejected. "
                "Contact the administrator.",
                "warning"
            )

        else:
            login_user(user)
            return redirect(url_for(dashboard_for(user)))

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have signed out.", "success")
    return redirect(url_for("home"))


# ---------------------------------------------------------
# STUDENT DASHBOARD AND ATTENDANCE
# ---------------------------------------------------------

@app.route("/student")
@role_required("student")
def student_dashboard():
    enrollments = Enrollment.query.filter_by(
        student_id=current_user.id
    ).all()

    active = (
        ClassSession.query
        .join(Enrollment, Enrollment.course_id == ClassSession.course_id)
        .filter(
            Enrollment.student_id == current_user.id,
            ClassSession.status == "open"
        )
        .order_by(
            ClassSession.session_date,
            ClassSession.start_time
        )
        .all()
    )

    stats = {
        enrollment.course_id: attendance_summary(
            current_user.id, enrollment.course_id
        )[0]
        for enrollment in enrollments
    }

    return render_template(
        "student_dashboard.html",
        enrollments=enrollments,
        active=active,
        stats=stats
    )


@app.route("/attendance/mark/<int:session_id>", methods=["POST"])
@role_required("student")
def mark_attendance(session_id):
    session = db.get_or_404(ClassSession, session_id)

    enrolled = Enrollment.query.filter_by(
        student_id=current_user.id,
        course_id=session.course_id
    ).first()

    if not enrolled or session.status != "open":
        abort(403)

    existing = Attendance.query.filter_by(
        session_id=session.id,
        student_id=current_user.id
    ).first()

    if existing:
        flash(
            "You have already submitted attendance for this class.",
            "info"
        )
    else:
        db.session.add(
            Attendance(
                session_id=session.id,
                student_id=current_user.id,
                status="pending",
                marked_at=datetime.utcnow()
            )
        )
        db.session.commit()

        flash(
            "Attendance submitted. It will count once the lecturer confirms it.",
            "success"
        )

    return redirect(url_for("student_dashboard"))


# ---------------------------------------------------------
# LECTURER DASHBOARD AND CLASS SCHEDULING
# ---------------------------------------------------------

@app.route("/lecturer")
@role_required("lecturer")
def lecturer_dashboard():
    courses = Course.query.filter_by(
        lecturer_id=current_user.id
    ).order_by(Course.name).all()

    sessions = (
        ClassSession.query
        .filter_by(lecturer_id=current_user.id)
        .order_by(
            ClassSession.session_date.desc(),
            ClassSession.start_time.desc()
        )
        .limit(60)
        .all()
    )

    return render_template(
        "lecturer_dashboard.html",
        courses=courses,
        sessions=sessions
    )


@app.route("/lecturer/schedule", methods=["GET", "POST"])
@role_required("lecturer")
def schedule_classes():
    courses = Course.query.filter_by(
        lecturer_id=current_user.id
    ).all()

    if request.method == "POST":
        try:
            course_id = int(request.form["course_id"])

            course = Course.query.filter_by(
                id=course_id,
                lecturer_id=current_user.id
            ).first_or_404()

            weekday = int(request.form["weekday"])
            start_t = time.fromisoformat(request.form["start_time"])
            end_t = time.fromisoformat(request.form["end_time"])
            first = date.fromisoformat(request.form["first_date"])
            last = date.fromisoformat(request.form["last_date"])
            room = request.form.get("room", "").strip()

            if (
                weekday not in range(7)
                or start_t >= end_t
                or last < first
                or (last - first).days > 370
            ):
                raise ValueError(
                    "Check the weekday, times, and date range "
                    "(maximum one year)."
                )

            schedule = Schedule(
                lecturer_id=current_user.id,
                course_id=course.id,
                weekday=weekday,
                start_time=start_t,
                end_time=end_t,
                first_date=first,
                last_date=last,
                room=room
            )

            db.session.add(schedule)
            db.session.flush()

            day = first
            while day <= last:
                if day.weekday() == weekday:
                    db.session.add(
                        ClassSession(
                            schedule_id=schedule.id,
                            course_id=course.id,
                            lecturer_id=current_user.id,
                            session_date=day,
                            start_time=start_t,
                            end_time=end_t,
                            room=room,
                            status="scheduled"
                        )
                    )

                day += timedelta(days=1)

            db.session.commit()
            flash(
                "Recurring timetable and class sessions created.",
                "success"
            )
            return redirect(url_for("lecturer_dashboard"))

        except (ValueError, KeyError) as exc:
            db.session.rollback()
            flash(
                str(exc) or "Enter valid schedule details.",
                "danger"
            )

    return render_template(
        "schedule.html",
        courses=courses,
        weekdays=list(enumerate(calendar.day_name))
    )


@app.route(
    "/lecturer/session/<int:session_id>/<action>",
    methods=["POST"]
)
@role_required("lecturer")
def manage_session(session_id, action):
    session = ClassSession.query.filter_by(
        id=session_id,
        lecturer_id=current_user.id
    ).first_or_404()

    if action == "open" and session.status == "scheduled":
        session.status = "open"
        session.opened_at = datetime.utcnow()
        db.session.commit()

        flash("Attendance register opened.", "success")

    elif action == "close" and session.status == "open":
        # Pending claims become absent when the register is closed.
        for record in Attendance.query.filter_by(
            session_id=session.id,
            status="pending"
        ).all():
            record.status = "absent"
            record.reviewed_at = datetime.utcnow()
            record.reviewed_by_id = current_user.id

        # Missing attendance submissions also become absent.
        enrolled_ids = [
            enrollment.student_id
            for enrollment in Enrollment.query.filter_by(
                course_id=session.course_id
            ).all()
        ]

        existing_ids = {
            attendance.student_id
            for attendance in Attendance.query.filter_by(
                session_id=session.id
            ).all()
        }

        for student_id in enrolled_ids:
            if student_id not in existing_ids:
                db.session.add(
                    Attendance(
                        session_id=session.id,
                        student_id=student_id,
                        status="absent",
                        reviewed_at=datetime.utcnow(),
                        reviewed_by_id=current_user.id
                    )
                )

        session.status = "closed"
        session.closed_at = datetime.utcnow()
        db.session.commit()

        flash(
            "Register closed. Unconfirmed claims and missing marks count as absent.",
            "success"
        )

    else:
        flash(
            "That session cannot perform the requested action.",
            "warning"
        )

    return redirect(
        url_for("attendance_register", session_id=session.id)
    )


@app.route("/lecturer/register/<int:session_id>")
@role_required("lecturer")
def attendance_register(session_id):
    session = ClassSession.query.filter_by(
        id=session_id,
        lecturer_id=current_user.id
    ).first_or_404()

    enrolled = Enrollment.query.filter_by(
        course_id=session.course_id
    ).all()

    records = {
        attendance.student_id: attendance
        for attendance in Attendance.query.filter_by(
            session_id=session.id
        ).all()
    }

    rows = [
        (enrollment.student, records.get(enrollment.student_id))
        for enrollment in enrolled
    ]

    return render_template(
        "attendance_register.html",
        session=session,
        rows=rows
    )


@app.route(
    "/lecturer/attendance/<int:attendance_id>/<decision>",
    methods=["POST"]
)
@role_required("lecturer")
def review_attendance(attendance_id, decision):
    record = db.get_or_404(Attendance, attendance_id)

    if record.session.lecturer_id != current_user.id:
        abort(403)

    if (
        record.session.status != "open"
        or record.status != "pending"
        or decision not in ("confirm", "reject")
    ):
        flash(
            "This attendance record can no longer be reviewed.",
            "warning"
        )
    else:
        record.status = (
            "present" if decision == "confirm" else "rejected"
        )
        record.reviewed_at = datetime.utcnow()
        record.reviewed_by_id = current_user.id
        db.session.commit()

        flash("Attendance record updated.", "success")

    return redirect(
        url_for(
            "attendance_register",
            session_id=record.session_id
        )
    )


# ---------------------------------------------------------
# CLASS ADVISOR DASHBOARD AND PREDICTION
# ---------------------------------------------------------

@app.route("/advisor")
@role_required("advisor")
def advisor_dashboard():
    return render_template(
        "advisor_dashboard.html",
        courses=Course.query.order_by(Course.name).all()
    )


@app.route("/predict", methods=["GET", "POST"])
@role_required("advisor")
def predict():
    students = (
        User.query
        .filter_by(role="student", status="approved")
        .order_by(User.student_id)
        .all()
    )

    courses = Course.query.order_by(Course.name).all()

    if request.method == "POST":
        try:
            student_id = int(request.form["student_user_id"])
            course_id = int(request.form["course_id"])

            student = User.query.filter_by(
                id=student_id,
                role="student",
                status="approved"
            ).first_or_404()

            course = db.get_or_404(Course, course_id)

            enrollment = Enrollment.query.filter_by(
                student_id=student.id,
                course_id=course.id
            ).first()

            if not enrollment:
                abort(
                    400,
                    "The selected student is not enrolled in this course."
                )

            assignment = float(request.form["assignment"])
            practical = float(request.form["practical"])
            ca = float(request.form["ca"])
            cgpa = float(request.form["cgpa"])
            participation = request.form["participation"]

            if participation not in PARTICIPATION_MAPPING:
                raise ValueError("Choose a valid participation level.")

            # Validate before storing the academic record.
            if not 0 <= assignment <= 20:
                raise ValueError(
                    "Assignment score must be between 0 and 20."
                )
            if not 0 <= practical <= 20:
                raise ValueError(
                    "Practical score must be between 0 and 20."
                )
            if not 0 <= ca <= 20:
                raise ValueError("CA score must be between 0 and 20.")
            if not 0 <= cgpa <= 5:
                raise ValueError(
                    "Previous CGPA must be between 0 and 5."
                )

            record = ensure_academic_record(student.id, course.id)
            record.assignment = assignment
            record.practical = practical
            record.ca = ca
            record.previous_cgpa = cgpa
            record.participation = participation
            db.session.commit()

            attendance, present, total = attendance_summary(
                student.id, course.id
            )

            prediction, confidence = get_prediction(
                attendance,
                assignment,
                practical,
                ca,
                cgpa,
                participation
            )

            return render_template(
                "result.html",
                student=student,
                course=course,
                prediction=prediction,
                confidence=confidence,
                attendance=attendance,
                present=present,
                total=total,
                assignment=assignment,
                practical=practical,
                ca=ca,
                cgpa=cgpa,
                participation=participation
            )

        except (ValueError, KeyError, RuntimeError) as exc:
            db.session.rollback()
            flash(
                str(exc) or "Enter valid prediction data.",
                "danger"
            )

    selected_student = request.args.get(
        "student_user_id", type=int
    )
    selected_course = request.args.get("course_id", type=int)

    record = None
    if selected_student and selected_course:
        record = AcademicRecord.query.filter_by(
            student_id=selected_student,
            course_id=selected_course
        ).first()

    return render_template(
        "predict.html",
        students=students,
        courses=courses,
        levels=LEVELS,
        selected_student=selected_student,
        selected_course=selected_course,
        record=record
    )


# ---------------------------------------------------------
# RESEND INTERVENTION EMAIL
# ---------------------------------------------------------

@app.route("/notify-lecturer", methods=["POST"])
@role_required("advisor")
def notify_lecturer():
    """
    Recalculate the current prediction on the server.
    Send an email only when the latest result is At Risk.
    """

    try:
        student_user_id = int(request.form["student_user_id"])
        course_id = int(request.form["course_id"])
    except (ValueError, KeyError):
        flash(
            "The student or course information is invalid.",
            "danger"
        )
        return redirect(url_for("advisor_dashboard"))

    student = User.query.filter_by(
        id=student_user_id,
        role="student",
        status="approved"
    ).first()

    course = db.session.get(Course, course_id)

    if student is None or course is None:
        flash(
            "The selected student or course could not be found.",
            "danger"
        )
        return redirect(url_for("advisor_dashboard"))

    enrollment = Enrollment.query.filter_by(
        student_id=student.id,
        course_id=course.id
    ).first()

    if enrollment is None:
        flash(
            "The selected student is not enrolled in this course.",
            "danger"
        )
        return redirect(url_for("advisor_dashboard"))

    record = AcademicRecord.query.filter_by(
        student_id=student.id,
        course_id=course.id
    ).first()

    if record is None:
        flash(
            "No academic record exists for this student and course. "
            "Run a prediction first.",
            "warning"
        )
        return redirect(url_for("advisor_dashboard"))

    lecturer = course.lecturer

    if lecturer is None or not lecturer.email:
        flash(
            "No lecturer with an email is assigned to this course. "
            "Ask the administrator to assign one.",
            "danger"
        )
        return redirect(url_for("advisor_dashboard"))

    if not resend.api_key:
        flash(
            "Email is not configured: RESEND_API_KEY is missing "
            "from your .env file.",
            "danger"
        )
        return redirect(url_for("advisor_dashboard"))

    if not MAIL_FROM:
        flash(
            "Email is not configured: MAIL_FROM is missing "
            "from your .env file.",
            "danger"
        )
        return redirect(url_for("advisor_dashboard"))

    try:
        attendance, present, total = attendance_summary(
            student.id, course.id
        )

        # Never trust a prediction or confidence value from the browser.
        prediction, confidence = get_prediction(
            attendance,
            record.assignment,
            record.practical,
            record.ca,
            record.previous_cgpa,
            record.participation
        )

        if prediction != "At Risk":
            flash(
                "A notification can only be sent when the current "
                "prediction is At Risk.",
                "warning"
            )
            return redirect(url_for("advisor_dashboard"))

        # Escape user-controlled values before inserting them into HTML.
        safe_lecturer_name = html.escape(lecturer.full_name)
        safe_student_id = html.escape(student.student_id or "Not provided")
        safe_course_name = html.escape(course.name)

        message_html = f"""
        <div style="font-family:Arial,sans-serif;max-width:650px;
                    margin:auto;padding:30px;color:#1e293b">
            <h2>Student Performance Intervention Alert</h2>

            <p>Dear {safe_lecturer_name},</p>

            <p>
                The Student Performance Prediction System has identified
                a student who may benefit from early academic support.
            </p>

            <div style="background:#f8fafc;padding:20px;border-radius:10px">
                <p><strong>Student ID:</strong> {safe_student_id}</p>
                <p><strong>Course:</strong> {safe_course_name}</p>
                <p><strong>Prediction:</strong> {html.escape(prediction)}</p>
                <p><strong>Model Confidence:</strong> {confidence}%</p>
                <p><strong>Attendance:</strong> {attendance}%</p>
                <p><strong>Confirmed Sessions Present:</strong> {present}</p>
                <p><strong>Finalized Sessions:</strong> {total}</p>
            </div>

            <p>
                Please review the student's academic situation and
                provide appropriate intervention where necessary.
            </p>

            <p>
                Regards,<br>
                <strong>Class Advisor</strong><br>
                Student Performance Prediction System
            </p>
        </div>
        """

        # Use a test recipient only when explicitly configured.
        # Otherwise send to the lecturer assigned to this course.
        recipient = RESEND_TEST_RECIPIENT or lecturer.email

        email_payload = {
            "from": MAIL_FROM,
            "to": [recipient],
            "subject": f"Student Performance Alert - {course.name}",
            "html": message_html
        }

        # The email send must remain inside this try block.
        resend.Emails.send(email_payload)

        app.logger.info(
    "Intervention email accepted by Resend for course ID %s.",
    course.id
)

        return render_template(
            "notification_success.html",
          student=student,
          course=course,
          lecturer=lecturer
         )

    except Exception:
        # The full exception is logged in the Flask terminal.
        # Do not display API keys or raw provider errors to users.
        app.logger.exception("Resend email sending failed")

        flash(
            "Could not send the email. Check the Flask terminal "
            "for the exact Resend error.",
            "danger"
        )
        return redirect(url_for("advisor_dashboard"))


# ---------------------------------------------------------
# ADMIN DASHBOARD, APPROVALS AND COURSE ASSIGNMENT
# ---------------------------------------------------------

@app.route("/admin")
@role_required("admin")
def admin_dashboard():
    pending = (
        User.query
        .filter(
            User.role.in_(["lecturer", "advisor"]),
            User.status == "pending"
        )
        .order_by(User.full_name)
        .all()
    )

    lecturers = (
        User.query
        .filter_by(role="lecturer", status="approved")
        .order_by(User.full_name)
        .all()
    )

    courses = Course.query.order_by(Course.name).all()
    students_count = User.query.filter_by(role="student").count()

    return render_template(
        "admin_dashboard.html",
        pending=pending,
        lecturers=lecturers,
        courses=courses,
        students=students_count
    )


@app.route(
    "/admin/user/<int:user_id>/<decision>",
    methods=["POST"]
)
@role_required("admin")
def admin_review_user(user_id, decision):
    user = db.get_or_404(User, user_id)

    if (
        user.role not in ("lecturer", "advisor")
        or user.status != "pending"
        or decision not in ("approve", "reject")
    ):
        abort(400)

    user.status = (
        "approved" if decision == "approve" else "rejected"
    )
    db.session.commit()

    flash(
        f"Account {decision}d.",
        "success"
    )
    return redirect(url_for("admin_dashboard"))


@app.route(
    "/admin/course/<int:course_id>/assign",
    methods=["POST"]
)
@role_required("admin")
def admin_assign_course(course_id):
    course = db.get_or_404(Course, course_id)
    lecturer_id = request.form.get("lecturer_id", type=int)

    if lecturer_id:
        lecturer = User.query.filter_by(
            id=lecturer_id,
            role="lecturer",
            status="approved"
        ).first()

        if lecturer is None:
            abort(400, "Choose an approved lecturer.")

        course.lecturer_id = lecturer.id
    else:
        course.lecturer_id = None

    db.session.commit()

    flash(
        f"Lecturer assignment updated for {course.name}.",
        "success"
    )
    return redirect(url_for("admin_dashboard"))


# ---------------------------------------------------------
# ERROR HANDLERS
# ---------------------------------------------------------

@app.errorhandler(403)
def forbidden(_error):
    return render_template(
        "error.html",
        code=403,
        message="You do not have permission to access this page."
    ), 403


@app.errorhandler(404)
def not_found(_error):
    return render_template(
        "error.html",
        code=404,
        message="The requested page or record could not be found."
    ), 404


# ---------------------------------------------------------
# LOCAL DEVELOPMENT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )
