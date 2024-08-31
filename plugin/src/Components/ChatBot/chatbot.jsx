import React, { useState, useEffect } from 'react';
import { FaRegMessage } from "react-icons/fa6";
import { IoMdSend } from "react-icons/io";
import { BsRobot } from "react-icons/bs";
import { AiOutlineLoading3Quarters } from "react-icons/ai"; // Importing loading icon
import axios from 'axios';
import './chatbot.css';

const Chatbot = ({ username, isAuthenticated }) => {
    const [showChat, setShowChat] = useState(false);
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState("");
    const [inputHeight, setInputHeight] = useState("40px");
    const [isLoading, setIsLoading] = useState(false);

    const toggleChat = () => {
        setShowChat(!showChat);
    };

    const handleInputChange = (e) => {
        setInputValue(e.target.value);
    };

    const handleSend = async () => {
        if (inputValue.trim() !== "") {
            const newMessages = [...messages, { role: 'user', content: inputValue }];
            setMessages(newMessages);
            setInputValue("");
            setInputHeight("40px");
            setIsLoading(true);

            try {
                const response = await axios.post('http://localhost:5000/chat', { message: inputValue, username }, { withCredentials: true });
                const botMessage = response.data.message;
                setMessages([...newMessages, { role: 'assistant', content: botMessage }]);
            } catch (error) {
                console.error('Error communicating with ChatGPT API:', error);
            } finally {
                setIsLoading(false);
            }
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            setInputHeight("80px");
            e.preventDefault();
            handleSend();
        }
    };

    const getInitials = (name) => {
        if (!name) return '';
        return name.split(' ').map((part) => part[0]).join('').toUpperCase();
    };

    useEffect(() => {
        const fetchChatHistory = async () => {
            try {
                const response = await axios.get('http://localhost:5000/chats', { withCredentials: true });
                if (response.status === 200) {
                    setMessages(response.data.flatMap(chat => chat.messages));
                }
            } catch (error) {
                console.error('Error fetching chat history:', error);
            }
        };

        if (isAuthenticated) {
            fetchChatHistory();
        }

        const handleStressAlert = (event) => {
            const { detail } = event;
            if (detail === 'stressed') {
                const alertMessage = `
                    Hi ${username}, it seems you are experiencing stress. Here are some tips to help you manage:
                    1. Take regular breaks throughout the day to refresh your mind and body.
                    2. Engage in physical activities such as walking or yoga to release endorphins and lift your mood.
                    3. Practice mindfulness and relaxation techniques like meditation and deep breathing exercises to calm your mind.
                    4. Maintain a healthy diet and stay hydrated to support your overall well-being.
                    5. Ensure you get enough sleep each night to feel rested and alert.
                    6. Please contact your family doctor for further guidance.
                    If you need further assistance, feel free to ask me!
                `;
                setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: alertMessage }]);
                setShowChat(true); // Automatically open chat on stress alert
            }
        };

        window.addEventListener('stressAlert', handleStressAlert);

        return () => {
            window.removeEventListener('stressAlert', handleStressAlert);
        };
    }, [username, isAuthenticated]);

    return (
        <div>
            {isAuthenticated && (
                <div className='message' onClick={toggleChat}>
                    <FaRegMessage className='message-icon' />
                </div>
            )}
            {showChat && (
                <div className='chat-ui'>
                    <div className='chat-header'>
                        Stress Bot
                        <button className='close-chat' onClick={toggleChat}>✖</button>
                    </div>
                    <div className='chat-body'>
                        {messages.map((msg, index) => (
                            <div key={index} className={`chat-message ${msg.role}`}>
                                {msg.role === 'user' ? (
                                    <div className="avatar">{getInitials(username)}</div>
                                ) : (
                                    <div className="avatar bot-avatar">
                                        <BsRobot className="bot-icon" />
                                    </div>
                                )}
                                <div className="message-content">
                                    {msg.content}
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className='chat-message assistant'>
                                <div className="avatar bot-avatar">
                                    <BsRobot className="bot-icon" />
                                </div>
                                <div className="message-content">
                                    <AiOutlineLoading3Quarters className="loading-icon" />
                                </div>
                            </div>
                        )}
                    </div>
                    <div className='chat-footer'>
                        <textarea 
                            placeholder='Type a message...'
                            value={inputValue}
                            onChange={handleInputChange}
                            onKeyPress={handleKeyPress}
                            style={{ height: inputHeight }}
                        />
                        <IoMdSend className='send-icon' onClick={handleSend} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default Chatbot;
