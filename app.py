from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

df = None


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/visualization")
def visualization():
    return render_template("visualization.html")


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/reports")
def reports():
    return render_template("reports.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# -----------------------------
# Upload Dataset
# -----------------------------
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    global df

    file = request.files["dataset"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    file.save(filepath)

    df = pd.read_csv(filepath)

    return render_template(
        "dataset.html",
        total_students=df.shape[0],
        total_features=df.shape[1],
        missing=df.isnull().sum().sum(),
        duplicate=df.duplicated().sum()
    )


# -----------------------------
# View Dataset
# -----------------------------
@app.route("/view_dataset")
def view_dataset():
    global df

    if df is None:
        return "Please upload a dataset first."

    table = df.head(100).to_html(index=False)

    return render_template("view_dataset.html", table=table)
@app.route("/dataset_summary")
def dataset_summary():
    global df

    if df is None:
        return "Please upload a dataset first."

    summary = df.describe(include="all").fillna("").to_html(index=True)

    return render_template("dataset_summary.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)