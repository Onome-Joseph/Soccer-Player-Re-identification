#  Soccer Player Re-Identification System
https://github.com/user-attachments/assets/7ea57982-1cae-432c-9c9d-0209c150cb00

This project performs real-time **player detection**, **tracking**, and **re-identification** in soccer videos. It combines:

- `YOLOv11` for object detection (players, goalkeepers, ball)
- `DeepSORT` for tracking across frames
- `TorchReID` (OSNet package) for appearance-based re-identification

Ellipse-style annotations are used for a video-game-like visual effect.

---

##  Features

- Detects players, goalkeepers, and ball from video feed
- Tracks players even during occlusions or exits/entries
- Matches player identities using appearance features (512-dim embeddings)

---
## Installation & Setup
#### 1. Clone the Repository
```
git clone https://github.com/Onome-Joseph/Soccer-Player-Re-identification.git
cd Soccer-Player-Re-identification
```
#### 2. Create Virtual Environment 
```
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```
#### 3. Install Requirements
```
pip install -r requirements.txt
```
#### 4. Running the Code
```
python reid_main.py
```

