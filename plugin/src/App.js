import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import Home from './Components/Home/home';
import axios from 'axios';
import Login from './Components/Login and Signup/login';
import Signup from './Components/Login and Signup/signup';
import Dashboard from './Components/Dashboard/dashboard';
import CustomNavbar from './Components/Navbar/navbar';
import Chatbot from './Components/ChatBot/chatbot';
import Footer from './Components/Footer/footer';
import About from './Components/About/about';
import Detector from './Components/Detector/detector';

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState('');
  const [message, setMessage] = useState('');

  const checkSession = async () => {
    try {
      const response = await axios.get('http://localhost:5000/@me', { withCredentials: true });
      console.log(response);
      if (response.status === 200 && response.data.message === 'Session active') {
        setIsAuthenticated(true);
        setUsername(response.data.username);
        setMessage(response.data.message);
      } else {
        setIsAuthenticated(false);
      }
    } catch (error) {
      console.error('Session check failed:', error);
      setIsAuthenticated(false);
    }
  };

  const handleLogin = async (username, password) => {
    try {
      const response = await axios.post('http://localhost:5000/login', { username, password }, { withCredentials: true });
      if (response.status === 200) {
        setIsAuthenticated(true);
        setUsername(username);
        checkSession();
        return true; // Indicate successful login
      } else {
        throw new Error('Login failed');
      }
    } catch (error) {
      throw error;
    }
  };

  useEffect(() => {
    checkSession();
  }, []);

  return (
    <Router>
      <div className="App">
        <CustomNavbar
          isAuthenticated={isAuthenticated}
          username={username}
          setIsAuthenticated={setIsAuthenticated}
          setUsername={setUsername}
          setmessage={setMessage}
        />
        <Routes>
          <Route path="/login" element={
            <Login 
              isAuthenticated={isAuthenticated} 
              setIsAuthenticated={setIsAuthenticated} 
              setUsername={setUsername} 
              onLogin={handleLogin} 
            />
          }/>
          <Route path="/signup" element={
            <Signup 
              setIsAuthenticated={setIsAuthenticated} 
              setUsername={setUsername} 
            />
          }/>
          <Route path="/home" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/about" element ={<About isAuthenticated={isAuthenticated} />} />
          <Route path="/detector" element={<Detector />} />
          <Route path="/" element={<Navigate to="/home" />} />
        </Routes>
      </div>
      <Chatbot username={username} message={message} isAuthenticated={isAuthenticated} />
      <Footer/>
    </Router>
  );
};

export default App;
