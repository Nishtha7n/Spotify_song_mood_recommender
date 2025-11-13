from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open("mood_classifier.pkl", "rb") as f:
    data = pickle.load(f)

clf = data["model"]
scaler = data["scaler"]
feature_cols = data["features"]

df = pd.read_csv("spotify_dataset.csv")

def get_song_features(song_name, artist_name):
    song_row = df[df["track_name"].str.lower().str.strip() == song_name.lower().strip()]

    if not song_row.empty:
        song_row = song_row[song_row["artists"].str.lower().str.contains(artist_name.lower().strip(), na=False)]

    if song_row.empty:
        return None

    features = song_row[feature_cols].values
    scaled_features = scaler.transform(features)
    return scaled_features[0], song_row

def recommend_songs(input_features, n_recommendations=5):
    # Scale all songs' features first
    all_features_scaled = scaler.transform(df[feature_cols])
    similarities = cosine_similarity([input_features], all_features_scaled)[0]

    # Get top N recommendations (excluding the same song)
    top_indices = similarities.argsort()[-(n_recommendations + 1):][::-1]
    recommended = df.iloc[top_indices][["track_name", "artists"]].reset_index(drop=True)

    return recommended.iloc[1:].to_dict(orient="records")  # skip the same song itself


@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_req = request.get_json()
        if not data_req or "song_name" not in data_req or "artist_name" not in data_req:
            return jsonify({"error": "Please provide 'song_name' and 'artist_name'"}), 400

        song_name = data_req["song_name"]
        artist_name = data_req["artist_name"]

        # Get song features
        result = get_song_features(song_name, artist_name)
        if result is None:
            return jsonify({"error": f"Song '{song_name}' by '{artist_name}' not found in dataset"}), 404

        features, song_row = result

        predicted_mood = clf.predict([features])[0]
        recommendations = recommend_songs(features)

        return jsonify({
            "song": song_name,
            "artist": artist_name,
            "predicted_mood": predicted_mood,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
