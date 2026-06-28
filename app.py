from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from flask import Flask, render_template, request, send_file
import os
import time
import csv
from werkzeug.utils import secure_filename
from src.predict import predict_full_document
from src.summarizer import generate_summary

def generate_pdf(filename, summary, prediction, processing_time):
    pdf_path = os.path.join("outputs", "ocr_report.pdf")

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>DocuVision AI OCR Report</b>", styles["Title"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(Paragraph(f"<b>Filename:</b> {filename}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Processing Time:</b> {processing_time} seconds", styles["Normal"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(Paragraph("<b>AI Summary</b>", styles["Heading2"]))
    elements.append(Paragraph(summary if summary else "No Summary", styles["BodyText"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(Paragraph("<b>Extracted Text</b>", styles["Heading2"]))
    elements.append(Paragraph(prediction.replace("\n", "<br/>"), styles["BodyText"]))

    doc.build(elements)

    return pdf_path


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = "outputs"
HISTORY_FILE = os.path.join(OUTPUT_FOLDER, "ocr_history.csv")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    preview_paths = None
    image_path = None
    prediction = None
    processing_time = None
    summary = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html", prediction="No image selected.")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", prediction="Please select an image.")

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        image_path = "/" + save_path.replace("\\", "/")

        try:
            start = time.time()

            prediction, preview_paths = predict_full_document(save_path)

            summary = generate_summary(prediction)

            pdf_path = generate_pdf(
    filename,
    summary,
    prediction,
    processing_time
)

            processing_time = round(time.time() - start, 2)

            file_exists = os.path.exists(HISTORY_FILE)

            with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["filename", "prediction", "summary", "processing_time"])

                writer.writerow([filename, prediction, summary, processing_time])

        except Exception as e:
            prediction = f"Error: {str(e)}"

        return render_template(
            "index.html",
            preview_paths=preview_paths,
            image_path=image_path,
            prediction=prediction,
            summary=summary,
            processing_time=processing_time
        )

    return render_template(
        "index.html",
        preview_paths=preview_paths,
        image_path=image_path,
        prediction=prediction,
        summary=summary,
        processing_time=processing_time
    )

@app.route("/download_history")
def download_history():

    if os.path.exists(HISTORY_FILE):
        return send_file(
            HISTORY_FILE,
            as_attachment=True,
            download_name="ocr_history.csv"
        )

    return "No OCR history available."
@app.route("/download_pdf")
def download_pdf():

    pdf_path = os.path.join("outputs", "ocr_report.pdf")

    if os.path.exists(pdf_path):
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name="ocr_report.pdf"
        )

    return "No PDF available."

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)