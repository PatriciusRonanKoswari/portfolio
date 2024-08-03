import React, { useState } from 'react';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../firebase';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async () => {
    if (!email || !password) {
      setError('Email and password are required.');
      return;
    }

    try {
      await signInWithEmailAndPassword(auth, email, password);
      // Redirect or update UI after login
      setError(''); // Clear any previous errors
    } catch (error) {
      console.error("Error signing in: ", error);
      if (error.code === 'auth/invalid-email') {
        setError('Invalid email address.');
      } else if (error.code === 'auth/wrong-password') {
        setError('Incorrect password.');
      } else {
        setError('An error occurred during login.');
      }
    }
  };

  return (
    <div>
      <h2 className='login-style'>LOGIN</h2>
      <input 
        type="email" 
        placeholder="Email" 
        value={email} 
        onChange={(e) => setEmail(e.target.value)} 
      />
      <input 
        type="password" 
        placeholder="Password" 
        value={password} 
        onChange={(e) => setPassword(e.target.value)} 
      />
      <button onClick={handleLogin} className='submit-btn'>Login</button>
      {error && <p className='error-message'>{error}</p>}
    </div>
  );
}

export default Login;
