import os
import numpy as np
import wave
import joblib
import librosa
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, login_required
from flask_session import Session
import logging
import openai
import requests
import speech_recognition as sr
import bcrypt
from datetime import datetime, timezone
from bson.objectid import ObjectId
from pydub import AudioSegment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'b53188295937de88acf7c26a8c0a9de11c915d7e1defe86ded74dbd07ef0c5e6'
app.config['MONGO_URI'] = 'mongodb+srv://jg581261:tubeligh@stressdetector.zkbcqby.mongodb.net/StressDetector?retryWrites=true&w=majority&appName=StressDetector'

# Use server-side session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = False
Session(app)

# OpenAI API key
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = 'sk-proj-2je0DedBHoockRuVvZHMT3BlbkFJLczGar5ppEYTd3f8NLBi'
openai.api_key = OPENAI_API_KEY

# Initialize MongoDB
mongo = PyMongo(app)

# Ensure unique index on username
mongo.db.users.create_index('username', unique=True)

# Check if MongoDB connection is successful
try:
    mongo.cx.server_info()
    print("Connected to MongoDB successfully")
except Exception as e:
    print("Failed to connect to MongoDB", e)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Ensure unauthorized users are redirected to login

# Configure CORS to allow credentials
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

# Configure logging
class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"{record.levelname}: {record.message}"

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())

app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Disable other loggers
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

logging.getLogger('pymongo').setLevel(logging.ERROR)

# Load the trained model and scaler
model = joblib.load('./recordings/model1.pkl')
scaler = joblib.load('./recordings/scaler1.pkl')

recognizer = sr.Recognizer()

class User(UserMixin):
    def __init__(self, user_id, username, password):
        self.id = user_id
        self.username = username
        self.password = password
@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(str(user['_id']), user['username'], user['password'])
    return None

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data['username']
    password = data['password']
    confirm_password = data['confirmPassword']
    
    if password != confirm_password:
        return jsonify({"message": "Passwords do not match"}), 400

    existing_user = mongo.db.users.find_one({'username': username})
    if existing_user:
        return jsonify({"message": "Username already exists"}), 400
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        user_id = mongo.db.users.insert_one({'username': username, 'password': hashed_password}).inserted_id
        app.logger.info(f"POST /register 201: User {username} ")
        return jsonify({"message": "User registered", "user_id": str(user_id)}), 201
    except Exception as e:
        app.logger.error(f"POST /register 500: Error during registration: {e}")
        return jsonify({"message": "Registration failed", "error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    try:
        user = mongo.db.users.find_one({"username": data['username']})
        if user and bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
            user_obj = User(str(user['_id']), user['username'], user['password'])
            login_user(user_obj)
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            app.logger.info(f"User logged in with username: {user['username']}")
            return jsonify({
                "message:": "Logged in",
                "id": str(user['_id']),
                "username": user['username']
            }), 200
        app.logger.warning("Invalid credentials")
        return jsonify({"message": "Invalid credentials"}), 401
    except Exception as e:
        app.logger.error(f"Error during login: {e}")
        return jsonify({"message": "Login failed", "error": str(e)}), 500

@app.route("/@me")
def get_current_user():
    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"message": "Not logged in"}), 401

    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    return jsonify({
        "id": str(user["_id"]),
        "username": user["username"],
        "message":"Session active"
    })

@app.route('/logout', methods=['POST'])
def logout_user():
    if 'user_id' in session:
        session.pop('user_id', None)
        app.logger.info("POST /logout 200: User logged out")
        return jsonify({"message": "Logged out"}), 200
    else:
        app.logger.warning("POST /logout 401: User not logged in")
        return jsonify({"message": "User not logged in"}), 401

@app.route('/update_username', methods=['POST'])
@login_required
def update_username():
    data = request.json
    new_username = data.get('new_username')
    user_id = session.get("user_id")

    if not new_username:
        return jsonify({"message": "New username is required"}), 400

    try:
        # Update the username in the database
        mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"username": new_username}}
        )
        # Also update the session username
        session['username'] = new_username
        app.logger.info(f"User {user_id} updated username to {new_username}")
        return jsonify({"message": "Username updated successfully", "new_username": new_username}), 200
    except Exception as e:
        app.logger.error(f"Error updating username: {e}")
        return jsonify({"message": "Failed to update username", "error": str(e)}), 500

def extract_features(y, sr):
    try:
        zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Calculate intensity as the mean amplitude of the signal
        intensity = np.mean(np.abs(y))
        
        # Calculate speech rate as the number of voiced frames per second
        voiced_frames = librosa.effects.split(y, top_db=20)
        speech_rate = len(voiced_frames) / (len(y) / sr)
        speech_rate = np.array([speech_rate])  # Ensure speech_rate is a 1D array

        # Calculate spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
        # Ensure the returned features match the expected dimensions
        features = np.hstack([
            zero_crossings, spectral_centroid, spectral_bandwidth, spectral_rolloff,
            mfccs_mean,
            chroma_mean,
            intensity,
            spectral_contrast_mean,
            speech_rate
        ])
        
        app.logger.info(f"Extracted features size: {features.size}")
        
        if features.size != 38:  # Ensure the features size matches the expected number
            raise ValueError(f"Expected 38 features, but got {features.size}")
        
        return features
    except Exception as e:
        app.logger.error(f"Error extracting features: {e}")
        return None

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    file = request.files['file']
    file_path = f"./recordings/{file.filename}"
    file.save(file_path)

    app.logger.info(f"Received file: {file.filename}")
    app.logger.info(f"File path: {file_path}")
    app.logger.info(f"File type: {file.content_type}")
    app.logger.info(f"File size: {os.path.getsize(file_path)} bytes")

    try:
        # Convert audio to ensure it is a valid WAV file
        audio = AudioSegment.from_file(file_path)
        audio.export(file_path, format="wav")

        # Load audio file and extract features
        y, sr = librosa.load(file_path, sr=None)

        # Define the number of frames (chunks) for the audio
        duration = librosa.get_duration(y=y, sr=sr)
        num_frames = 4 if duration <= 60 else int(duration // 15)

        # Split the audio into chunks
        chunk_size = len(y) // num_frames
        frame_predictions = []

        for i in range(num_frames):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            y_chunk = y[start:end]

            # Extract features for each chunk
            features = extract_features(y_chunk, sr)
            if features is None:
                return jsonify({"error": "Failed to extract features"}), 500

            if features.size != 38:
                return jsonify({"error": f"Expected 38 features, but got {features.size}"}), 500

            # Normalize the features
            normalized_features = scaler.transform([features])

            # Make a prediction for the chunk
            model_input = np.expand_dims(normalized_features, axis=2)
            model_prediction = model.predict(model_input).tolist()[0][0]
            frame_predictions.append(model_prediction)
            app.logger.info(f"Chunk {i + 1}/{num_frames} prediction: {model_prediction}")

        # Calculate the mean prediction
        mean_prediction = np.mean(frame_predictions)
        predicted_label = "stressed" if mean_prediction >= 0.23 else "not stressed"
        app.logger.info(f"Mean prediction: {mean_prediction}, Final label: {predicted_label}")
    
        reason = "Model classified the emotion."

        user_id = session.get("user_id")
        # Create a dictionary of feature names and their values
        feature_names = [
            'zero_crossings', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            *[f'mfcc_{i+1}' for i in range(13)],
            *[f'chroma_{i+1}' for i in range(12)],
            'intensity',
            *[f'spectral_contrast_{i+1}' for i in range(7)],
            'speech_rate'
        ]
        feature_values = {name: value for name, value in zip(feature_names, features)}

        # Add the prediction result to the user's record in the database
        mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"emotions": {
                "emotion": predicted_label,
                "reason": reason,
                "features": feature_values,
                "model_prediction": mean_prediction,
                "timestamp": datetime.now(timezone.utc)
            }}}
        )

        return jsonify({
            "emotion": predicted_label,
            "reason": reason,
            "features": feature_values,
            "model_prediction": mean_prediction
        })

    except RuntimeError as e:
        app.logger.error(f"Runtime error during audio processing: {e}")
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        app.logger.error(f"Error during audio processing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/user_emotions', methods=['GET'])
def get_user_emotions():
    user_id = session.get("user_id")
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "emotions": 1})
    
    if user and "emotions" in user:
        return jsonify(user["emotions"])
    else:
        return jsonify([]), 200
    
@app.route('/stress_dashboard', methods=['GET'])
def stress_dashboard():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"message": "Not logged in"}), 401

    user = mongo.db.users.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "emotions": 1})

    if user and "emotions" in user:
        stress_data = user["emotions"]
        return jsonify(stress_data), 200
    else:
        return jsonify([]), 200

class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"{record.levelname}: {record.message}"

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())

app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
    
@app.route('/chat', methods=['POST'])
@login_required
def chat_with_gpt():
    data = request.json
    user_message = data['message']
    user_id = session.get("user_id")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}]
        )
        message_content = response.choices[0].message['content']

        # Store chat in MongoDB
        mongo.db.chats.insert_one({
            "user_id": ObjectId(user_id),
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": message_content}
            ],
            "timestamp": datetime.now(timezone.utc)
        })

        return jsonify({"message": message_content}), 200
    except Exception as e:
        app.logger.error(f"Error communicating with ChatGPT API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chats', methods=['GET'])
@login_required
def get_chats():
    user_id = session.get("user_id")
    chats = mongo.db.chats.find({"user_id": ObjectId(user_id)})
    chat_list = [{"messages": chat["messages"], "timestamp": chat["timestamp"]} for chat in chats]
    return jsonify(chat_list), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
