# prediction.py
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('trained_mood_model.pkl')
scaler = joblib.load('scaler.pkl')

# Spotify API credentials
cid = "3aef8f1d3ef540fc9f677ab240127907"
secret = "f1988553f1744abeb9370ff3d7e91387"
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_audio_features(track_uri):
    """Retrieve audio features for a single track using its URI."""
    features = sp.audio_features(track_uri)[0]
    if features:
        feature_values = [
            features['danceability'], features['energy'], features['key'],
            features['loudness'], features['mode'], features['speechiness'],
            features['acousticness'], features['instrumentalness'],
            features['liveness'], features['valence'], features['tempo'],
            features['duration_ms'], features['time_signature']
        ]
        return feature_values
    else:
        return None

def predict_mood(track_uri):
    """Predict the mood of a song given its URI."""
    # Get audio features
    features = get_audio_features(track_uri)
    if features:
        # Scale the features
        features_scaled = scaler.transform([features])
        # Predict mood
        predicted_mood = model.predict(features_scaled)[0]
        return predicted_mood
    else:
        return "Could not retrieve features for prediction."
