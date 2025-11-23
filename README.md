# Baseball Speed Tracking

Upload your video to the website to find an estimate of how fast the baseball was thrown.
This program was done in Python, using YOLO v8 to track the ball. Panda was used to interpolate between missing frames, and to track the speed of the ball. The website will be run locally on your computer. Make sure your environment can display GUI from OpenCV.

# Installation

1. Set up your Python environment. Make sure you have installed Pip
    ```bash
    python -m venv myenv
    ```
2. Install the requirements.txt file.
    ```bash
    pip install -r requirements.txt
    ```
3. Run app.py
4. Open the website locally.

# How to Use

1. Make sure your video is high quality and the baseball is visible.
2. Select the frame where your ball is thrown.
3. Select the frame where your ball is caught.
4. Select the area of where the ball will likely be for the duration of the clip.
5. Draw a line from the batter to the pitcher or vice versa for the speed calculations.
6. Download the video if your environment doesn't support OpenCV GUI to see the result.

# Group Members 

1. Brandon Micheal
2. Khang Ngo
3. Srikar Andhavarapu
