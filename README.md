# 📄 DocuVision AI – Intelligent Document Digitization System

An AI-powered document digitization system that converts handwritten and scanned documents into editable digital text using **CRNN (Convolutional Recurrent Neural Network)** based Optical Character Recognition (OCR). The application provides an end-to-end pipeline from document preprocessing and line segmentation to text extraction, AI-powered summarization, and PDF report generation through a modern Flask web interface.

---

## 🚀 Features

* 📤 Upload handwritten or scanned document images
* 🖼️ Image preprocessing using OpenCV
* ⚫ Grayscale conversion
* ⚪ Adaptive thresholding for noise reduction
* ✂️ Automatic text line segmentation
* 🧠 CRNN-based OCR for handwritten text recognition
* 🤖 AI-powered document summarization using Google Gemini
* 🔍 Search within extracted text
* 📊 OCR statistics (Words, Characters, Lines)
* 📄 Download extracted text (.txt)
* 📑 Download OCR report (.pdf)
* 📈 OCR processing history (CSV)
* 📱 Responsive and modern user interface
* ⚡ Real-time OCR processing

---

# 🏗️ System Workflow

```text
Upload Document
       │
       ▼
Image Preprocessing
(Grayscale + Thresholding)
       │
       ▼
Text Line Segmentation
       │
       ▼
CRNN OCR Recognition
       │
       ▼
Extracted Text
       │
       ▼
AI Summary (Gemini)
       │
       ▼
Export Report (TXT / PDF)
```

---

# 🧠 OCR Pipeline

The OCR pipeline consists of the following stages:

1. Document Upload
2. Image Preprocessing
3. Text Line Segmentation
4. CRNN-based Text Recognition
5. CTC Decoding
6. AI Summary Generation
7. Export Results

---

# 🛠️ Technology Stack

## Backend

* Python
* Flask

## Deep Learning

* PyTorch
* CRNN (Convolutional Recurrent Neural Network)
* CTC Loss

## Computer Vision

* OpenCV

## AI

* Google Gemini API

## Data Processing

* Pandas
* NumPy

## PDF Generation

* ReportLab

## Frontend

* HTML5
* CSS3
* Bootstrap 5
* JavaScript

---

# 📂 Project Structure

```text
document_digitization_OCR/
│
├── app.py
├── requirements.txt
├── .env
│
├── checkpoints/
├── data/
│
├── outputs/
│   ├── cropped_lines/
│   ├── ocr_history.csv
│   └── ocr_report.pdf
│
├── src/
│   ├── model.py
│   ├── predict.py
│   ├── segment.py
│   ├── summarizer.py
│   └── document_classifier.py
│
├── static/
│   ├── previews/
│   └── uploads/
│
└── templates/
    └── index.html
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/DocuVision-AI.git
```

Move into the project directory

```bash
cd DocuVision-AI
```

Create a virtual environment

```bash
conda create -n crnn_env python=3.10
```

Activate the environment

```bash
conda activate crnn_env
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
GEMINI_API_KEY=YOUR_API_KEY
```

Run the application

```bash
python app.py
```

Open

```text
http://127.0.0.1:5000
```

---

# 📊 Application Features

* Upload document image
* Automatic OCR processing
* AI-generated document summary
* Processing time display
* OCR text search
* Word, character, and line statistics
* Download extracted text
* Download PDF report
* OCR history logging

---

# 📷 Application Screens

* Home Page
* OCR Pipeline
* Upload Interface
* Image Preprocessing
* Line Segmentation
* AI OCR Result
* AI Document Summary
* PDF Report Export


---

# 📈 Future Enhancements

* Multi-language OCR support
* OCR confidence score visualization
* Document type classification
* Cloud deployment
* Batch document processing
* User authentication
* Database integration
* REST API support

---

# 🎯 Applications

* Digital Archiving
* Historical Document Preservation
* Educational Notes Digitization
* Office Automation
* Research Document Processing
* Healthcare Record Digitization
* Government Document Management

---



