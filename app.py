from flask import Flask, render_template, request, flash, redirect, url_for
import os
import re
from object_track2 import process_video

app = Flask(__name__)
app.secret_key = "baseball_tracker_secret_key_2024"
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

        # get real distance from form
        real_distance_meters = float(request.form.get("distance", 18.44))  # default ~18.44 m

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, "processed_" + file.filename)

        file.save(input_path)
        try:
            # pass the distance to process_video
            summary = process_video(input_path, output_path, real_distance_meters)
        except Exception as e:
            return f"An error occurred while processing the video: {e}", 500

        return render_template("index.html",
                               processed_video=output_path,
                               summary=summary)

    return render_template("index.html", processed_video=None)


if __name__ == "__main__":
    app.run(debug=True)
