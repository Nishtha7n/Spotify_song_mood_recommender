# Spotify Mood Prediction

This project predicts the **mood of a song** (Happy, Sad, Energetic, Calm) based on its audio features such as tempo, energy, loudness, and valence.

---

## Project Structure

SPOTIFY_PROJECT/
│
├── templates/
│ └── home.html                    # Web interface
│
├── venv/                          # Virtual environment
├── spotify_dataset.csv            # Dataset
├── mood_classifier.pkl            # Trained model
├── scaling.pkl                    # Optional scaler
├── app.py                         # Flask backend
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation