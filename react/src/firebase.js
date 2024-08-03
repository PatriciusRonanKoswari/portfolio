// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
// Add other Firebase products you want to use (optional)
// import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBOHn-1IblBKiJ7Yzlhozwo9N9B_vV5B2k",
  authDomain: "cakra-project-54f76.firebaseapp.com",
  projectId: "cakra-project-54f76",
  storageBucket: "cakra-project-54f76.appspot.com",
  messagingSenderId: "575584387401",
  appId: "1:575584387401:web:1358b6912e688bb10a5a98",
  measurementId: "G-7DBV25RDQN"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);

// Initialize Cloud Firestore and get a reference to the service
export const firestore = getFirestore(app);

// Optional: Initialize Analytics and get a reference to the service
// const analytics = getAnalytics(app);
