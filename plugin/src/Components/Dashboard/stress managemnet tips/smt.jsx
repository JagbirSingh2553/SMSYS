import React, { useState } from 'react';
import { FaWalking, FaSpa, FaLeaf, FaAppleAlt, FaBed } from 'react-icons/fa';
import { IoIosArrowBack, IoIosArrowForward } from 'react-icons/io';
import './smt.css';
import exercise1Gif from '../../../assets/yoga.gif'; // Import your GIFs
import exercise2Gif from '../../../assets/break.gif';
import exercise3Gif from '../../../assets/food.gif';

const exercises = [
  {
    img: exercise1Gif,
    title: 'Exercise 1: Deep Breathing',
    description: 'Doing this will help to calm your mind and body.'
  },
  {
    img: exercise2Gif,
    title: 'Exercise 2: Take a Break',
    description: 'Take a break from work which causes stress.'
  },
  {
    img: exercise3Gif,
    title: 'Exercise 3: Do thing which makes you happy',
    description: 'Some people like to eat when there is stress, or some people like to travel Or some take a nap.'
  }
];

const StressManagementTips = () => {
  const [showPopup, setShowPopup] = useState(false);
  const [currentExercise, setCurrentExercise] = useState(0);

  const togglePopup = () => {
    setShowPopup(!showPopup);
  };

  const nextExercise = () => {
    setCurrentExercise((prev) => (prev + 1) % exercises.length);
  };

  const prevExercise = () => {
    setCurrentExercise((prev) => (prev - 1 + exercises.length) % exercises.length);
  };

  return (
    <div className="stress-management-tips-section">
      <h3>Stress Management Tips</h3>
      <ul>
        <li>
          <i><FaWalking /></i>
          <span>
            <div>Take regular breaks throughout the day.</div>
            <div>This helps to reduce stress and improve productivity. Short breaks can refresh your mind and body.</div>
          </span>
        </li>
        <li>
          <i><FaSpa /></i>
          <span>
            <div>Engage in physical activities such as walking or yoga.</div>
            <div>Physical activities can help manage stress. Exercise releases endorphins, which are natural mood lifters.</div>
          </span>
        </li>
        <li>
          <i><FaLeaf /></i>
          <span>
            <div>Practice mindfulness and relaxation techniques.</div>
            <div>Techniques like meditation can help calm your mind. Deep breathing exercises can reduce anxiety.</div>
          </span>
        </li>
        <li>
          <i><FaAppleAlt /></i>
          <span>
            <div>Maintain a healthy diet and stay hydrated.</div>
            <div>A good diet and hydration are essential for managing stress. Eating balanced meals can improve your overall mood.</div>
          </span>
        </li>
        <li>
          <i><FaBed /></i>
          <span>
            <div>Ensure you get enough sleep each night.</div>
            <div>Proper sleep is crucial for stress management. Aim for 7-9 hours of sleep each night to feel rested and alert.</div>
          </span>
        </li>
      </ul>
      <button className="btn smt-btn" onClick={togglePopup}>Try Relaxation Exercise</button>
      {showPopup && (
        <div className="popup">
          <div className="popup-inner">
            <h4>Relaxation Exercises</h4>
            <div className="exercise-card-container">
              <button className="arrow-btn" onClick={prevExercise}><IoIosArrowBack /></button>
              <div className="card">
                <img src={exercises[currentExercise].img} alt={exercises[currentExercise].title} className="exercise-gif" />
                <div>{exercises[currentExercise].title}</div>
                <p>{exercises[currentExercise].description}</p>
              </div>
              <button className="arrow-btn" onClick={nextExercise}><IoIosArrowForward /></button>
            </div>
            <button className="btn close-btn" onClick={togglePopup}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default StressManagementTips;
