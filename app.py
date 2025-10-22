from flask import Flask, render_template, request, send_from_directory
import os
from object_track import process_video

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return "No file uploaded", 400

        file = request.files["video"]
        if file.filename == "":
            return "No selected file", 400

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, "processed_" + file.filename)

        file.save(input_path)
        process_video(input_path, output_path)

        return render_template("index.html", processed_video=output_path)

    return render_template("index.html", processed_video=None)

if __name__ == "__main__":
    app.run(debug=True)
