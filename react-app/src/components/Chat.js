import React, { useState, useEffect } from 'react';
import { firestore } from '../firebase';
import { collection, query, orderBy, onSnapshot, addDoc } from 'firebase/firestore';
import EmojiPicker from 'emoji-picker-react';

function Chat() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);

  useEffect(() => {
    const messagesCollection = collection(firestore, 'messages');
    const q = query(messagesCollection, orderBy('timestamp'));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const messagesData = snapshot.docs.map(doc => doc.data());
      setMessages(messagesData);
    });
    return () => unsubscribe();
  }, []);

  const handleSendMessage = async () => {
    try {
      const messagesCollection = collection(firestore, 'messages');
      await addDoc(messagesCollection, {
        text: message,
        timestamp: new Date(),
      });
      setMessage('');
    } catch (error) {
      console.error("Error sending message: ", error);
    }
  };

  const onEmojiClick = (emojiData, event) => {
    console.log('Emoji Data:', emojiData);
    if (emojiData && emojiData.emoji) {
      setMessage(prevMessage => prevMessage + emojiData.emoji);
    } else {
      console.log("Emoji object does not have the expected emoji property.");
    }
    setShowEmojiPicker(false);
  };

  return (
    <div>
      <div>
        {messages.map((msg, index) => (
          <div key={index}>{msg.text}</div>
        ))}
      </div>
      <button className='open-emoji' onClick={() => setShowEmojiPicker(!showEmojiPicker)}>
        {showEmojiPicker ? 'Close Emoji Picker' : 'Open Emoji Picker'}
      </button>
      {showEmojiPicker && (
        <EmojiPicker onEmojiClick={onEmojiClick} />
      )}
      <input 
        type="text" 
        value={message} 
        onChange={(e) => setMessage(e.target.value)} 
      />
      <button onClick={handleSendMessage}>Send</button>
    </div>
  );
}

export default Chat;
