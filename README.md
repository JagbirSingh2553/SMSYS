Stress Detection System
Overview

The Stress Detection System is a comprehensive web application designed to detect stress through real-time voice analysis. This system leverages advanced machine learning models to analyze audio input, determine stress levels, and provide timely feedback and stress management tips. The goal is to offer a self-monitoring tool that can help users manage their stress proactively.
Features

    Real-time Stress Detection: Analyzes voice recordings in real-time to detect stress.
    User Authentication: Secure login and registration using Flask, MongoDB, and session management.
    Data Storage: Stores user emotion history in MongoDB for future analysis and tracking.
    Notifications: Real-time alerts when stress is detected, integrated directly into the browser.
    Dashboard: A user-friendly dashboard to monitor stress levels over time.
    Interactive Chatbot: Integrates with OpenAI's GPT-4 to offer stress management advice and support.
    Browser Extension: The application is also available as a Chrome extension for seamless stress monitoring during browsing sessions.

Technology Stack
Frontend

    React.js: For building dynamic user interfaces.
    HTML/CSS/JavaScript: Standard web technologies for creating a responsive and interactive UI.
    Chrome Extension API: Used to integrate the application into the browser, allowing for real-time notifications and interaction.

Backend

    Flask: A lightweight Python web framework used for handling API requests and server-side logic.
    MongoDB: A NoSQL database used to store user data and emotion history.
    TensorFlow & Librosa: Libraries used for machine learning model training and audio feature extraction.

Installation
Prerequisites

    Backend: Python 3.7+
    Frontend: Node.js and React.js
    Database: MongoDB (local or cloud instance)
    Browser: Google Chrome for the extension

Clone the Repository

bash

git clone https://github.com/yourusername/stress-detection-system.git
cd stress-detection-system

Backend Setup

    Install Dependencies:

    bash

pip install -r requirements.txt

Environment Variables: Set up environment variables for MongoDB URI, OpenAI API Key, etc.

Run the Backend:

bash

    python testbackend.py

Frontend Setup (Chrome Extension)

    Unzip the ChromeExtension.zip and navigate to the folder:

    bash

unzip ChromeExtension.zip
cd ChromeExtension

Install Dependencies:

bash

npm install

Build the Extension:

bash

    npm run build

    Load the Extension in Chrome:
        Open Chrome and go to chrome://extensions/.
        Enable "Developer mode" in the top right corner.
        Click on "Load unpacked" and select the build folder from the ChromeExtension directory.

Running the Application

    Backend: Ensure the Flask server is running by executing python testbackend.py.
    Frontend: The Chrome extension should now be loaded in your browser. Click on the extension icon to start using the stress detection features.

Usage
Stress Detection

    Voice Recording: Users can record their voice, and the system will analyze the recording to detect stress.
    Real-Time Notifications: Receive alerts when stress is detected.

Dashboard

    Monitor History: View historical data of stress levels, which are stored in MongoDB.

Interactive Chatbot

    Stress Management: Users can interact with the chatbot to receive advice on managing stress.

Model Details

    Model Architecture: The model is a Recurrent Neural Network (RNN) with LSTM layers, processing sequential data. It utilizes audio features like MFCCs, chroma, and spectral contrast to predict stress levels.
    Training: The model was trained on audio data with data augmentation techniques such as noise addition and pitch shifting.
    Performance: Achieves an accuracy of approximately 82.34% on the test set.

API Endpoints
Authentication

    POST /register: Register a new user.
    POST /login: Login an existing user.
    POST /logout: Logout the current user.

Stress Detection

    POST /predict_emotion: Predicts the stress level from the uploaded audio file.

User Data

    GET /@me: Get the current logged-in user's data.
    GET /user_emotions: Retrieve the logged-in user's emotion history.

Chatbot

    POST /chat: Interact with the GPT-4 powered chatbot for stress management advice.
    GET /chats: Retrieve past chat interactions.

Future Work

    Enhanced Stress Detection: Improving the model to detect varying levels of stress.
    Noise Robustness: Optimizing the model for performance in noisy environments.
    Mobile Compatibility: Extending the application’s compatibility to mobile platforms.

License

This project is licensed under the MIT License - see the LICENSE file for details.
