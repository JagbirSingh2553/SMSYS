import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './detector.css'; // Importing the CSS file

const Detector = ({ username }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [emotion, setEmotion] = useState('');
    const [details, setDetails] = useState('');
    const [userEmotions, setUserEmotions] = useState([]);
    const [filter, setFilter] = useState('All');
    const [interval, setIntervalTime] = useState(1); // Default 1 minute in minutes
    const [feedback, setFeedback] = useState(''); // State to track feedback
    const mediaRecorderRef = useRef(null);
    const intervalRef = useRef(null);
    const isRecordingActive = useRef(false); // Use useRef to manage recording state
    const audioContextRef = useRef(null); // Use ref to manage AudioContext
    const streamRef = useRef(null); // Reference to the stream

    const fetchUserEmotions = async () => {
        try {
            const response = await axios.get('http://localhost:5000/user_emotions', { withCredentials: true });
            console.log("Fetched user emotions:", response.data); // Log backend response
            setUserEmotions(response.data);
        } catch (error) {
            console.error("Error fetching user emotions:", error);
        }
    };

    useEffect(() => {
        fetchUserEmotions();
        // Check the recording status when the component is mounted
        chrome.runtime.sendMessage({ action: 'getStatus' }, (response) => {
            setIsRecording(response.isRecording);
        });
        // Clear the badge when the popup is opened
        chrome.runtime.sendMessage({ action: 'clearBadge' }, (response) => {
            console.log(response.status); // Log the response for debugging
        });
    }, []);

    const startRecording = async () => {
        console.log(username);
        setFeedback('Recording...'); // Set feedback message to "Recording..."

        try {
            // Request access to the user's microphone
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream; // Store stream reference

            // Initialize the AudioContext for pitch visualization
            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            }

            const analyser = audioContextRef.current.createAnalyser();
            const source = audioContextRef.current.createMediaStreamSource(stream);
            source.connect(analyser);

            const canvas = document.getElementById('pitchCanvas');
            const canvasContext = canvas.getContext('2d');

            analyser.fftSize = 2048;
            const bufferLength = analyser.fftSize;
            const dataArray = new Uint8Array(bufferLength);

            // Set recording state to active
            isRecordingActive.current = true;

            const drawPitch = () => {
                if (!isRecordingActive.current) return; // Stop drawing if recording is stopped
                requestAnimationFrame(drawPitch);

                analyser.getByteTimeDomainData(dataArray);

                // Set canvas background color
                canvasContext.fillStyle = '#3A4A6B';
                canvasContext.fillRect(0, 0, canvas.width, canvas.height);

                // Set waveform color
                canvasContext.lineWidth = 2;
                canvasContext.strokeStyle = '#B0C4DE';

                canvasContext.beginPath();

                const sliceWidth = canvas.width * 1.0 / bufferLength;
                let x = 0;

                for (let i = 0; i < bufferLength; i++) {
                    const v = dataArray[i] / 128.0;
                    const y = v * canvas.height / 2;

                    if (i === 0) {
                        canvasContext.moveTo(x, y);
                    } else {
                        canvasContext.lineTo(x, y);
                    }

                    x += sliceWidth;
                }

                canvasContext.lineTo(canvas.width, canvas.height / 2);
                canvasContext.stroke();
            };

            drawPitch(); // Start visualizing the pitch

            // Start recording using MediaRecorder
            mediaRecorderRef.current = new MediaRecorder(stream);

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    const blob = new Blob([event.data], { type: 'audio/wav' });

                    // Send the audio chunk to the backend for emotion prediction
                    processAudioPrediction(blob);
                }
            };

            mediaRecorderRef.current.start();
            console.log('MediaRecorder started');

            // Set interval to split the recording into chunks based on the selected interval
            intervalRef.current = setInterval(() => {
                if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
                    mediaRecorderRef.current.stop(); // Stop current recording
                    mediaRecorderRef.current.start(); // Immediately start a new recording
                }
            }, interval * 60 * 1000); // Convert minutes to milliseconds

            setIsRecording(true);
            chrome.runtime.sendMessage({ action: 'startRecording' });
        } catch (error) {
            console.error('Error accessing microphone', error);
        }
    };

    const stopRecording = () => {
        clearInterval(intervalRef.current); // Clear the interval
        isRecordingActive.current = false; // Stop the visualization

        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop(); // Stop the final recording
        }

        setIsRecording(false);
        setFeedback('Recording stopped.'); // Update feedback to "Recording stopped."

        // Close the AudioContext and release microphone stream
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null; // Reset the AudioContext reference
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop()); // Stop all media tracks
            streamRef.current = null;
        }

        chrome.runtime.sendMessage({ action: 'stopRecording' });
    };

    const processAudioPrediction = async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'recording.wav');

        try {
            const response = await axios.post('http://localhost:5000/predict_emotion', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                withCredentials: true
            });

            const detectedEmotion = response.data.emotion;
            const emotionDetails = response.data.reason;

            setEmotion(`Detected Emotion: ${detectedEmotion === 'stressed' ? 'Stressed' : 'Not Stressed'}`);
            setDetails(`Details: ${emotionDetails}`);

            // Fetch historical emotions after new prediction is made
            fetchUserEmotions();

            if (detectedEmotion === 'stressed') {
                const event = new CustomEvent('stressAlert', { detail: 'stressed' });
                window.dispatchEvent(event);
                chrome.runtime.sendMessage({ action: 'stressDetected' });
            } else {
                chrome.runtime.sendMessage({ action: 'clearBadge' });
            }

            // Second API request (which I didn't remove)
            const secondApiResponse = await axios.post('http://localhost:5000/another_endpoint', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                withCredentials: true
            });
            console.log('Second API response:', secondApiResponse.data);

        } catch (error) {
            console.error('Error uploading the file', error);
        }
    };

    const handleFilterChange = (event) => {
        setFilter(event.target.value);
    };

    const handleIntervalChange = (e) => {
        setIntervalTime(Number(e.target.value)); // Set interval in minutes
    };

    const filteredEmotions = userEmotions.filter(entry => {
        if (filter === 'All') return true;
        if (filter === 'Stressed') return entry.emotion === 'stressed';
        if (filter === 'Not Stressed') return entry.emotion === 'not stressed';
        return false;
    });

    return (
        <>
            <div className="dropdown-container">
                <div>
                    <label className="dropdown-label">Select Interval (in minutes):</label>
                    <input type="number" min="1" defaultValue="1" placeholder='Default value is 1 Minute' onChange={handleIntervalChange} className="select-dropdown" />
                </div>
            </div>
            <div className="btn-container">
                <button className="success" onClick={startRecording} disabled={isRecording}>Start</button>
                <button className="danger" onClick={stopRecording} disabled={!isRecording}>Stop</button>
            </div>
            <canvas id="pitchCanvas" width="600" height="100" style={{ border: '1px solid #007bff', marginTop: '20px', backgroundColor: '#3A4A6B' }}></canvas> {/* Pitch visual canvas */}
            <div>
                {emotion && <p className="emotion">{emotion}</p>}
                {details && <p className="details">{details}</p>}
                {feedback && (
                    <p className={`feedback ${!isRecording ? 'feedback-stopped' : ''}`}>
                        {feedback}
                    </p>
                )}
            </div>
            <div className="filter-container">
                <label className="filter-label">Filter Results:</label>
                <select value={filter} onChange={handleFilterChange} className="select-dropdown">
                    <option value="All">All</option>
                    <option value="Stressed">Stressed</option>
                    <option value="Not Stressed">Not Stressed</option>
                </select>
            </div>
            <div className="emotion">
                <h2>Previous Results</h2>
                <div className="results-container">
                    <ul>
                        {filteredEmotions.map((entry, index) => (
                            <li key={index}>
                                {new Date(entry.timestamp).toLocaleString()}: {entry.emotion === 'stressed' ? 'Stressed' : 'Not Stressed'}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </>
    );
};

export default Detector;
