from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


# Load the trained Machine Learning model
model = joblib.load("student_performance_model.pkl")


# Convert participation values into the same
# numerical values used during model training
PARTICIPATION_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}


# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")


# ABOUT PAGE
@app.route("/about")
def about():
    return render_template("about.html")


# PREDICTION PAGE
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:

            # Student Information

            level = request.form["level"]


            # Attendance Information

            classes_held = int(
                request.form["classes_held"]
            )

            classes_attended = int(
                request.form["classes_attended"]
            )


            # Validate attendance

            if classes_held <= 0:

                return """
                <h3>Error</h3>
                <p>Number of classes held must be greater than zero.</p>
                <a href="/predict">Go Back</a>
                """


            if classes_attended < 0:

                return """
                <h3>Error</h3>
                <p>Classes attended cannot be negative.</p>
                <a href="/predict">Go Back</a>
                """


            if classes_attended > classes_held:

                return """
                <h3>Error</h3>
                <p>
                Classes attended cannot be greater than
                classes held.
                </p>
                <a href="/predict">Go Back</a>
                """


            # Calculate attendance automatically

            attendance = (
                classes_attended / classes_held
            ) * 100

            attendance = round(attendance, 2)


            # Academic Information

            assignment = float(
                request.form["assignment"]
            )

            practical = float(
                request.form["practical"]
            )

            ca = float(
                request.form["ca"]
            )

            cgpa = float(
                request.form["cgpa"]
            )

            participation = request.form[
                "participation"
            ]


            # Validate Scores

            if not 0 <= assignment <= 20:

                return """
                <h3>Error</h3>
                <p>Assignment score must be between 0 and 20.</p>
                <a href="/predict">Go Back</a>
                """


            if not 0 <= practical <= 20:

                return """
                <h3>Error</h3>
                <p>Practical score must be between 0 and 20.</p>
                <a href="/predict">Go Back</a>
                """


            if not 0 <= ca <= 20:

                return """
                <h3>Error</h3>
                <p>CA score must be between 0 and 20.</p>
                <a href="/predict">Go Back</a>
                """


            if not 0 <= cgpa <= 5:

                return """
                <h3>Error</h3>
                <p>CGPA must be between 0 and 5.</p>
                <a href="/predict">Go Back</a>
                """


            # Encode Participation

            participation_value = (
                PARTICIPATION_MAPPING[
                    participation
                ]
            )


            # Prepare Model Input

            input_data = pd.DataFrame(
                [[
                    attendance,
                    assignment,
                    practical,
                    ca,
                    cgpa,
                    participation_value
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


            # Make Prediction

            prediction = model.predict(
                input_data
            )[0]


            # Prediction Probability

            probabilities = (
                model.predict_proba(
                    input_data
                )[0]
            )

            confidence = round(
                float(max(probabilities)) * 100,
                2
            )


            # Display Result

            return render_template(
                "result.html",

                prediction=prediction,

                confidence=confidence,

                level=level,

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

            return f"""
            <h3>Error: Missing form field</h3>
            <p>{error}</p>
            <a href="/predict">Go Back</a>
            """


        except ValueError:

            return """
            <h3>Error</h3>
            <p>
            Please enter valid numbers in the
            numerical fields.
            </p>
            <a href="/predict">Go Back</a>
            """


    return render_template(
        "predict.html"
    )


# RUN APPLICATION

if __name__ == "__main__":
    app.run()