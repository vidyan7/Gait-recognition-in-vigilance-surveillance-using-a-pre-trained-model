# Gait-recognition-in-vigilance-surveillance-using-a-pre-trained-model

## Abstract
Gait recognition is a biometric technique that identifies individuals based on their walking patterns. 
This project proposes a gait recognition system for vigilance surveillance using a pre-trained pose estimation model. 
The system extracts skeletal landmarks from video frames and computes gait features to recognize individuals without relying on facial data.

## Problem Statement
Traditional surveillance systems depend heavily on facial recognition, which fails in low-resolution, occluded, or long-distance scenarios. 
There is a need for an alternative biometric approach that works effectively in such conditions.

## Objectives
- To design a gait-based biometric recognition system
- To utilize a pre-trained model for human pose estimation
- To extract and analyze gait features from video sequences
- To compare gait patterns for accurate identification
- To enhance vigilance surveillance systems

## Features
- Human pose estimation using **MediaPipe**
- Gait feature extraction (spatial, temporal, frequency & wavelet)
- Multiple similarity techniques:
  - Cosine Similarity
  - Dynamic Time Warping (DTW)
  - Pearson Correlation
  - Euclidean Distance
  - Manhattan Similarity
  - Feature-Weighted Similarity
- Real-time gait recognition using webcam
- Admin panel with authentication
- Profile comparison and PDF report generation

---

## 🛠️ Tech Stack
- **Python**
- **Flask**
- **OpenCV**
- **MediaPipe**
- **NumPy, SciPy**
- **Scikit-learn**
- **SQLite**
- **ReportLab**

---
## System Architecture
1. Video Input
2. Pose Estimation
3. Feature Extraction
4. Gait Analysis
5. Similarity Matching
6. Identification Output

## Project Structure
```
gait-recognition-system/
│
├── app.py                  # Main application file containing Flask routes and gait recognition logic
├── requirements.txt        # List of required Python libraries
├── README.md               # Project documentation
├── gait_forensics.db       # SQLite database (optional; not recommended to upload real data)
│
├── templates/              # HTML files for the web interface
│   ├── index.html
│   ├── admin_login.html
│   ├── profiles.html
│   ├── compare.html
│   ├── realtime.html
│   ├── identify_admin.html
│   └── view_profile.html
│
├── static/                 # CSS, JS, images, and other static assets
│   ├── assets/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│
├── uploads/                # Folder to store uploaded videos (keep empty)
│
└── .gitignore              # Files/folders to ignore when uploading to GitHub


```

## How to Run the Project
1. Install Python 3.8 or above
2. Install required libraries:
   pip install -r requirements.txt
3. Run the application:
   python app.py
4. Access the system at:
   http://127.0.0.1:5000/

## Results
The system successfully identifies individuals based on gait patterns. 
Experimental results demonstrate effective recognition under varying conditions such as lighting and distance.

## Applications
- Vigilance surveillance
- Security and monitoring systems
- Restricted area access control
- Smart city surveillance

## Future Enhancements
- Integration with deep learning-based gait models
- Support for large-scale gait databases
- Improved accuracy using temporal modeling
- Cloud-based deployment



## How to Run the Project

### 1️⃣ Clone the repository
Open your terminal (or Git Bash) and type:

```bash
git clone https://github.com/vidyan7/Gait-recognition-in-vigilance-surveillance-using-a-pre-trained-model.git
cd Gait-recognition-in-vigilance-surveillance-using-a-pre-trained-model
```
###2️⃣ Install required libraries

Make sure you have Python 3.8 or above installed.
Then install all required libraries using:
```
pip install -r requirements.txt
```
###3️⃣ Run the application

Start the Flask application by typing:
```
python app.py
```
The server will start and show output similar to:
```
* Running on http://127.0.0.1:4500/ (Press CTRL+C to quit)
```
