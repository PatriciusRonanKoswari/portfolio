import React from 'react';
import Login from './components/Login';
import Chat from './components/Chat';
import './App.css';


function App() {
  return (
    <div className="App">
      <h1>Chat Application</h1>
      <div className="bg-styling">
        <Login />
        <Chat />
      </div>
      
    </div>
  );
}

export default App;
