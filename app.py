from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import resend

from dotenv import load_dotenv
from course_lecturers import COURSE_LECTURERS


# =========================================
# LOAD ENVIRONMENT VARIABLES
# =========================================

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")


# =========================================
# FLASK APPLICATION
# =========================================

app = Flask(__name__)


# =========================================
# LOAD MACHINE LEARNING MODEL
# =========================================

model = joblib.load("student_performance_model.pkl")


# =========================================
# CLASS PARTICIPATION MAPPING
# =========================================

PARTICIPATION_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}


# =========================================
# HOME
# =========================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================
# ABOUT
# =========================================

@app.route("/about")
def about():
    return render_template("about.html")


# =========================================
# PREDICTION
# =========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:

            # -----------------------------
            # STUDENT INFORMATION
            # -----------------------------

            student_id = request.form["student_id"]
            level = request.form["level"]
            course = request.form["course"]

            # -----------------------------
            # ATTENDANCE
            # -----------------------------

            classes_held = int(request.form["classes_held"])
            classes_attended = int(request.form["classes_attended"])

            # -----------------------------
            # ACADEMIC PERFORMANCE
            # -----------------------------

            assignment = float(request.form["assignment"])
            practical = float(request.form["practical"])
            ca = float(request.form["ca"])

            # -----------------------------
            # PREVIOUS PERFORMANCE
            # -----------------------------

            cgpa = float(request.form["cgpa"])

            # -----------------------------
            # PARTICIPATION
            # -----------------------------

            participation = request.form["participation"]

            # -----------------------------
            # VALIDATE ATTENDANCE
            # -----------------------------

            if classes_held <= 0:
                return "Error: Number of classes held must be greater than 0."

            if classes_attended < 0:
                return "Error: Classes attended cannot be negative."

            if classes_attended > classes_held:
                return "Error: Classes attended cannot be greater than classes held."

            # -----------------------------
            # CALCULATE ATTENDANCE
            # -----------------------------

            attendance = (classes_attended / classes_held) * 100
            attendance = round(attendance, 2)

            # -----------------------------
            # VALIDATE SCORES
            # -----------------------------

            if not 0 <= assignment <= 20:
                return "Error: Assignment score must be between 0 and 20."

            if not 0 <= practical <= 20:
                return "Error: Practical score must be between 0 and 20."

            if not 0 <= ca <= 20:
                return "Error: CA score must be between 0 and 20."

            if not 0 <= cgpa <= 5:
                return "Error: CGPA must be between 0 and 5."

            # -----------------------------
            # VALIDATE PARTICIPATION
            # -----------------------------

            if participation not in PARTICIPATION_MAPPING:
                return "Error: Invalid class participation value."

            participation_value = PARTICIPATION_MAPPING[participation]

            # -----------------------------
            # CHECK COURSE (Keep this to get the lecturer's name)
            # -----------------------------

            lecturer = COURSE_LECTURERS.get(course)

            if lecturer is None:
                return "Error: No lecturer is assigned to this course."

            # -----------------------------
            # PREPARE MODEL INPUT
            # -----------------------------

            input_data = pd.DataFrame([[
                attendance,
                assignment,
                practical,
                ca,
                cgpa,
                participation_value
            ]], columns=[
                "Attendance",
                "Assignment_Score",
                "Practical_Score",
                "CA_Score",
                "Previous_CGPA",
                "Class_Participation"
            ])

            # -----------------------------
            # MAKE PREDICTION
            # -----------------------------

            prediction = model.predict(input_data)[0]

            # -----------------------------
            # CALCULATE MODEL CONFIDENCE
            # -----------------------------

            probabilities = model.predict_proba(input_data)[0]

            confidence = round(
                float(max(probabilities)) * 100,
                2
            )

            # -----------------------------
            # LECTURER INFORMATION
            # -----------------------------

            lecturer_name = lecturer["name"]

            # -----------------------------
            # DISPLAY RESULT
            # -----------------------------

            return render_template(
                "result.html",

                student_id=student_id,
                level=level,
                course=course,

                prediction=prediction,
                confidence=confidence,

                lecturer_name=lecturer_name,

                attendance=attendance,
                classes_held=classes_held,
                classes_attended=classes_attended,

                assignment=assignment,
                practical=practical,
                ca=ca,

                cgpa=cgpa,
                participation=participation
            )

        except KeyError as error:

            return f"Error: Missing form field: {error}"

        except ValueError:

            return "Error: Please enter valid numbers in the numeric fields."

    return render_template("predict.html")


# =========================================
# NOTIFY COURSE LECTURER
# =========================================

@app.route("/notify-lecturer", methods=["POST"])
def notify_lecturer():

    try:

        # -----------------------------
        # RECEIVE INFORMATION
        # -----------------------------

        student_id = request.form["student_id"]
        course = request.form["course"]
        prediction = request.form["prediction"]
        confidence = request.form["confidence"]
        
        # -----------------------------
        # GRAB TYPED EMAIL FROM WEB FORM
        # -----------------------------
        
        lecturer_email = request.form["lecturer_email"]

        # -----------------------------
        # CHECK COURSE (For Name Only)
        # -----------------------------

        lecturer = COURSE_LECTURERS.get(course)

        if lecturer is None:
            return "Error: No lecturer is assigned to this course."

        lecturer_name = lecturer["name"]

        # -----------------------------
        # ONLY ALLOW AT RISK ALERTS
        # -----------------------------

        if prediction != "At Risk":
            return "Notification is only available for students predicted to be At Risk."

        # -----------------------------
        # CHECK API KEY
        # -----------------------------

        if not resend.api_key:
            return "Error: Email service API key has not been configured."

        # -----------------------------
        # EMAIL CONTENT
        # -----------------------------

        email_parameters = {
            "from": "Student Performance System <alerts@eze.name.ng>",
            "to": [lecturer_email],
            "subject": (
                f"Student Performance Alert - {course}"
            ),

            "html": f"""
                <div style="
                    font-family: Arial, sans-serif;
                    max-width: 650px;
                    margin: auto;
                    padding: 30px;
                    color: #1e293b;
                ">

                    <h2 style="color: #0f172a;">
                        Student Performance Intervention Alert
                    </h2>

                    <p>
                        Dear {lecturer_name},
                    </p>

                    <p>
                        The Student Performance Prediction System
                        has identified a student who may require
                        early academic intervention.
                    </p>

                    <div style="
                        background: #f8fafc;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;
                    ">

                        <p>
                            <strong>Student ID:</strong>
                            {student_id}
                        </p>

                        <p>
                            <strong>Course:</strong>
                            {course}
                        </p>

                        <p>
                            <strong>Prediction:</strong>
                            {prediction}
                        </p>

                        <p>
                            <strong>Model Confidence:</strong>
                            {confidence}%
                        </p>

                    </div>

                    <p>
                        Please review the student's academic
                        situation and provide appropriate
                        intervention where necessary.
                    </p>

                    <p>
                        Regards,<br>
                        <strong>Class Advisor</strong><br>
                        Student Performance Prediction System
                    </p>

                </div>
            """
        }

        print(f"---- DEBUG: Sending email to: {lecturer_email} ----")

        # -----------------------------
        # SEND EMAIL 
        # -----------------------------

        resend.Emails.send(email_parameters)

        # -----------------------------
        # SHOW SUCCESS PAGE
        # -----------------------------

        return render_template(
            "notification_success.html",

            student_id=student_id,
            course=course,
            lecturer_name=lecturer_name
        )

    except Exception as error:

        return f"SYSTEM CHECK ERROR: {error}"


# APPLICATION START

if __name__ == "__main__":
    app.run()