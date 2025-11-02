from flask import Flask, render_template, request, flash, redirect, url_for
import os
import re
from object_track import process_video

app = Flask(__name__)
app.secret_key = "baseball_tracker_secret_key_2024"
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("video")
        if not file or file.filename == "":
            flash("Please select a video file to upload.", "error")
            return redirect(url_for("index"))

        # Validate file extension
        allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            flash(f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}", "error")
            return redirect(url_for("index"))

        try:
            # Sanitize filename to avoid URL encoding issues
            safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
            
            # Paths
            input_path = os.path.join(UPLOAD_FOLDER, safe_filename)
            output_filename = "processed_" + os.path.splitext(safe_filename)[0] + ".mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            # Save uploaded file
            file.save(input_path)
            
            # Process video with Object_trackv3 algorithm
            print(f"Processing video: {input_path}")
            process_video(input_path, output_path)
            
            # Check if output file was created
            if not os.path.exists(output_path):
                flash("Video processing failed. Please check the console for errors.", "error")
                return redirect(url_for("index"))

            # Render webpage with processed video
            relative_video_path = f"output/{output_filename}"
            flash("Video processed successfully! The baseball tracking algorithm has been applied.", "success")
            return render_template("index.html", processed_video=relative_video_path)
        
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            flash(f"An error occurred while processing the video: {str(e)}", "error")
            return redirect(url_for("index"))

    # Default GET
    return render_template("index.html", processed_video=None)


if __name__ == "__main__":
    app.run(debug=True)
