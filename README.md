## Sets Counter (Pose Classification with Computer Vision)

A computer vision–based rep counter that tracks exercises with clear up/down movement patterns (e.g., sit-ups, squats, toe touches, push-ups).
It uses MediaPipe Pose for body landmark detection, a trained classifier (deadlift.pkl) to recognize exercise stages, and a Tkinter + CustomTkinter GUI to display live feedback and stats.

✅ Counts reps in real-time
✅ Detects UP / DOWN stages automatically
✅ Toggle counting with a simple hand gesture (open palm)
✅ Displays stage, reps, probability, and status in a clean dashboard

### Features

🎥 Real-time webcam feed with pose overlay

🏋️ Exercise rep counting (any up/down based movement)

✋ Gesture control: Open palm to start/pause

🖥️ Simple, modern GUI (CustomTkinter)

📊 Info cards: Stage | Reps | Probability | Status

🔄 Reset button to restart counter

### Requirements

Install the dependencies:  
pip install -r requirements.txt

### Files in this repo

main.py → The main program (GUI + CV pipeline)

landmarks.py → Defines landmark names (required, copy as is)

deadlift.pkl → Pre-trained classifier model (used to detect up/down stages)

### Usage

1. Clone the repo:  
    git clone https://github.com/yourusername/sets-counter.git  
    cd sets-counter

2. Make sure landmarks.py and deadlift.pkl are in the same directory as main.py.

3. Run the app:  
    python app.py

4. The webcam feed will appear with pose tracking.

5. Show an open palm to toggle ACTIVE/PAUSED mode.

6. Perform your exercise — reps will be counted automatically.

7. Use the RESET button to reset your counter.

