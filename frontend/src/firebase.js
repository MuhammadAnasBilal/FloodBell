import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, signInWithEmailAndPassword, createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyC6RUcCjiulTnjphjVZ8YIDp4o1Ghr0EbM",
  authDomain: "flood-bell.firebaseapp.com",
  projectId: "flood-bell",
  storageBucket: "flood-bell.firebasestorage.app",
  messagingSenderId: "1015886062955",
  appId: "1:1015886062955:web:112017d4aed2454f41182e"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const googleProvider = new GoogleAuthProvider();

export { signInWithPopup, signInWithEmailAndPassword, createUserWithEmailAndPassword, updateProfile };
