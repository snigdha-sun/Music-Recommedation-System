import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the NLTK Sentiment Analyzer data
nltk.download('vader_lexicon')

# Load and preprocess data
data = pd.read_csv(r"C:\Users\Sindhu\OneDrive\Desktop\ML Project\spotify_cleaned.csv")

features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
target = 'mood'

# Generate mood column if not present
if target not in data.columns:
    def classify_mood(valence, energy):
        if valence > 0.7 and energy > 0.6:
            return 'Happy'
        elif valence <= 0.7 and energy > 0.7:
            return 'Energetic'
        elif valence >= 0.4 and energy <= 0.6:
            return 'Relaxed'
        else:
            return 'Sad'

    data[target] = data.apply(lambda row: classify_mood(row['valence'], row['energy']), axis=1)

# Encode target labels
label_encoder = LabelEncoder()
data['mood_encoded'] = label_encoder.fit_transform(data[target])

# Prepare features and target
X = data[features]
y = data['mood_encoded']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Map moods to songs
mood_to_songs = {}
for mood, songs in data.groupby(target)['track_name']:
    mood_to_songs[mood] = songs.tolist()

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to extract features from user input
def extract_features_from_input(user_input):
    sentiment_scores = sentiment_analyzer.polarity_scores(user_input)
    valence = (sentiment_scores['pos'] - sentiment_scores['neg'] + 1) / 2  
    energy = abs(sentiment_scores['compound'])
    return scaler.transform([[valence, energy, 0.5, 0.5, 120]])

def detect_mood(user_input):
    user_features = extract_features_from_input(user_input)
    predicted_mood_encoded = model.predict(user_features)[0]
    predicted_mood = label_encoder.inverse_transform([predicted_mood_encoded])[0]

    sentiment_scores = sentiment_analyzer.polarity_scores(user_input)
    compound_score = sentiment_scores['compound']

    keywords_to_moods = {
        'happy': 'Happy',
        'energetic': 'Energetic',
        'active': 'Energetic',
        'relaxed': 'Relaxed',
        'calm': 'Relaxed',
        'sad': 'Sad',
        'down': 'Sad',
        'melancholy': 'Sad'
    }

    for keyword, mood in keywords_to_moods.items():
        if keyword in user_input.lower():
            return mood

    if compound_score > 0.05:
        return 'Happy'
    elif compound_score < -0.05:
        return 'Sad'
    else:
        return 'Relaxed'

# Streamlit UI
st.title("Spotify Mood Detection and Song Recommendation")
st.write("Enter your feelings, and we'll recommend songs that match your mood!")

user_input = st.text_input("How are you feeling today?")

if user_input:
    detected_mood = detect_mood(user_input)

    if detected_mood in mood_to_songs:
        recommended_songs = mood_to_songs[detected_mood]

        st.write(f"### Detected Mood: {detected_mood}")
        st.write(f"### Recommended Songs for {detected_mood}:")
        for idx, song in enumerate(recommended_songs[:50], 1):
            st.write(f"{idx}. {song}")
    else:
        st.write("No songs available for the detected mood.")
