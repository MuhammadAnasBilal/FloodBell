import React, { useState, useEffect, useCallback } from 'react';
import { Mail, Lock, AlertCircle, Eye, EyeOff, User, Phone, Calendar, Moon, Sun, CloudRain } from 'lucide-react';
import { auth, db, googleProvider, signInWithPopup, signInWithEmailAndPassword, createUserWithEmailAndPassword, updateProfile } from '../firebase';
import { doc, setDoc } from 'firebase/firestore';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// --- Particle Components --- //
const RainDrop = ({ x, y, onHit }) => {
  useEffect(() => {
    // 600ms is the duration of the fall
    const timer = setTimeout(() => {
      onHit(x, window.innerHeight - 80);
    }, 500);
    return () => clearTimeout(timer);
  }, [x, y, onHit]);

  return (
    <motion.div
      initial={{ x, y, opacity: 0.8, scaleY: 1 }}
      animate={{ y: window.innerHeight, opacity: 0, scaleY: 3 }}
      transition={{ duration: 0.5, ease: "easeIn" }}
      className="fixed top-0 left-0 w-[2px] h-[20px] bg-blue-400 rounded-full z-0 pointer-events-none"
    />
  );
};

const Ripple = ({ x, y }) => {
  return (
    <motion.div
      initial={{ x: x - 15, y: y - 5, scale: 0, opacity: 0.8 }}
      animate={{ scale: 4, opacity: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="fixed top-0 left-0 w-[30px] h-[10px] border-2 border-blue-300 rounded-[50%] z-0 pointer-events-none"
    />
  );
};

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const [isDark, setIsDark] = useState(true);
  
  // Form State
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [phone, setPhone] = useState('');
  const [dob, setDob] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Mouse Cloud State
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000 });
  const [raindrops, setRaindrops] = useState([]);
  const [ripples, setRipples] = useState([]);
  const [isHoveringCard, setIsHoveringCard] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  // Track mouse
  useEffect(() => {
    const handleMouseMove = (e) => setMousePos({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Spawn rain
  useEffect(() => {
    const interval = setInterval(() => {
      if (isHoveringCard) return;

      const id = Math.random().toString(36).substr(2, 9);
      // Spawn within a 60px radius of the mouse to match the cloud width perfectly
      const offsetX = mousePos.x + (Math.random() * 80 - 40);
      const offsetY = mousePos.y + (Math.random() * 20);
      
      setRaindrops(prev => [...prev.slice(-30), { id, x: offsetX, y: offsetY }]);
    }, 50);
    return () => clearInterval(interval);
  }, [mousePos, isHoveringCard]);

  const handleRipple = useCallback((x, y) => {
    const id = Math.random().toString(36).substr(2, 9);
    setRipples(prev => [...prev.slice(-15), { id, x, y }]);
  }, []);

  const handleGoogleSignIn = async () => {
    try {
      setLoading(true);
      setError('');
      const userCredential = await signInWithPopup(auth, googleProvider);
      
      // Fire-and-forget Firestore save so it doesn't block login if Firestore is unconfigured
      setDoc(doc(db, "users", userCredential.user.uid), {
        email: userCredential.user.email,
        fullName: userCredential.user.displayName,
        createdAt: new Date().toISOString(),
      }, { merge: true }).catch(err => console.warn("Firestore save failed:", err));

      navigate('/dashboard');
    } catch (err) {
      handleFirebaseError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleEmailAuth = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setError('');
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const fullName = `${firstName} ${lastName}`;
        await updateProfile(userCredential.user, { displayName: fullName });
        
        // Fire-and-forget Firestore save so it doesn't block login if Firestore is unconfigured
        setDoc(doc(db, "users", userCredential.user.uid), {
          firstName,
          lastName,
          fullName,
          phone,
          dob,
          email,
          createdAt: new Date().toISOString()
        }).catch(err => console.warn("Firestore save failed:", err));
      }
      navigate('/dashboard');
    } catch (err) {
      handleFirebaseError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleFirebaseError = (err) => {
    console.error(err);
    if (err.code === 'auth/configuration-not-found') {
      setError("Firebase Authentication is not fully configured.");
    } else if (err.code === 'auth/invalid-credential') {
      setError("Incorrect email or password. Please try again.");
    } else if (err.code === 'auth/email-already-in-use') {
      setError("An account already exists with this email address.");
    } else {
      setError(err.message || "An error occurred during authentication.");
    }
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1], staggerChildren: 0.05 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }
  };

  const isLight = !isDark;
  const inputBgClass = isLight ? "bg-white border-gray-300 text-gray-800 placeholder-gray-500" : "bg-black/40 border-white/10 text-white placeholder-gray-500";
  const labelClass = isLight ? "text-gray-600" : "text-gray-400";
  const iconClass = isLight ? "text-gray-400" : "text-gray-500";

  // Dynamic wave colors to prevent being too bright in light mode
  const waveColors = isDark 
    ? ["rgba(14, 165, 233, 0.7)", "rgba(59, 130, 246, 0.5)", "rgba(29, 78, 216, 0.3)", "rgba(30, 58, 138, 1)"]
    : ["rgba(14, 165, 233, 0.4)", "rgba(59, 130, 246, 0.3)", "rgba(29, 78, 216, 0.2)", "rgba(30, 58, 138, 0.4)"];

  return (
    <div className="min-h-screen flex items-center justify-center bg-brand-dark px-4 relative overflow-hidden transition-colors duration-500">
      
      {/* Theme Toggle Button */}
      <motion.button
        whileHover={{ scale: 1.1, rotate: isDark ? 90 : -90 }}
        whileTap={{ scale: 0.9 }}
        onClick={() => setIsDark(!isDark)}
        className={`absolute top-6 right-6 z-50 p-3 rounded-full ${isLight ? 'bg-white shadow-xl border-gray-200' : 'bg-brand-card shadow-lg border-white/10'} text-brand-primary`}
      >
        <AnimatePresence mode="wait">
          {isDark ? (
            <motion.div key="moon" initial={{ opacity: 0, rotate: -90 }} animate={{ opacity: 1, rotate: 0 }} exit={{ opacity: 0, rotate: 90 }}>
              <Sun size={24} className="text-yellow-400" />
            </motion.div>
          ) : (
            <motion.div key="sun" initial={{ opacity: 0, rotate: -90 }} animate={{ opacity: 1, rotate: 0 }} exit={{ opacity: 0, rotate: 90 }}>
              <Moon size={24} className="text-blue-600" />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.button>

      {/* Realistic Mist Cloud Cursor (Top Left Fixed to fix alignment issue) */}
      <motion.div
        animate={{ 
          x: mousePos.x - 75, // Center the 150px icon precisely 
          y: mousePos.y - 75,
          opacity: isHoveringCard ? 0 : 1 // Hide when hovering card
        }}
        transition={{ type: 'tween', ease: 'easeOut', duration: 0.1 }}
        className="fixed top-0 left-0 z-40 pointer-events-none flex items-center justify-center"
        style={{ width: '150px', height: '150px' }}
      >
        {/* Layering multiple clouds with blur to create a volumetric, realistic mist effect rather than flat cartoon */}
        <CloudRain size={160} className={`absolute ${isDark ? 'text-slate-400' : 'text-slate-600'} drop-shadow-2xl blur-[4px] opacity-80`} />
        <CloudRain size={140} className={`absolute ${isDark ? 'text-gray-300' : 'text-gray-500'} blur-[1px] opacity-90`} />
      </motion.div>

      {/* Render Particles */}
      {!isHoveringCard && raindrops.map(drop => (
        <RainDrop key={drop.id} x={drop.x} y={drop.y} onHit={handleRipple} />
      ))}
      {ripples.map(rip => (
        <Ripple key={rip.id} x={rip.x} y={rip.y} />
      ))}

      {/* Animated Flood Waves Background */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden opacity-30">
        <svg className="absolute bottom-0 w-full h-[30vh]" viewBox="0 24 150 28" preserveAspectRatio="none">
          <defs>
            <path id="gentle-wave" d="M-160 44c30 0 58-18 88-18s 58 18 88 18 58-18 88-18 58 18 88 18 v44h-352z" />
          </defs>
          <g className="animate-[wave_10s_linear_infinite]">
            <use href="#gentle-wave" x="48" y="0" fill={waveColors[0]} className="transition-colors duration-500" />
          </g>
          <g className="animate-[wave_15s_linear_infinite_reverse]">
            <use href="#gentle-wave" x="48" y="3" fill={waveColors[1]} className="transition-colors duration-500" />
          </g>
          <g className="animate-[wave_20s_linear_infinite]">
            <use href="#gentle-wave" x="48" y="5" fill={waveColors[2]} className="transition-colors duration-500" />
          </g>
          <g className="animate-[wave_25s_linear_infinite_reverse]">
            <use href="#gentle-wave" x="48" y="7" fill={waveColors[3]} className="transition-colors duration-500" />
          </g>
        </svg>
      </div>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes wave {
          0% { transform: translateX(0) }
          100% { transform: translateX(-176px) }
        }
      `}} />

      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        onMouseEnter={() => setIsHoveringCard(true)}
        onMouseLeave={() => setIsHoveringCard(false)}
        className={`w-full max-w-[500px] bg-brand-card p-6 md:p-8 rounded-[2rem] border ${isLight ? 'border-gray-200' : 'border-white/10'} shadow-[0_0_50px_rgba(0,0,0,0.2)] backdrop-blur-2xl relative z-50 overflow-hidden transition-colors duration-500`}
      >
        {/* Branding */}
        <motion.div variants={itemVariants} className="flex flex-col items-center mb-6 relative z-10">
          <motion.div 
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.5 }}
            className={`w-20 h-20 ${isLight ? 'bg-slate-800' : 'bg-white/5'} rounded-3xl flex items-center justify-center mb-3 overflow-hidden shadow-xl transition-colors duration-500`}
          >
            <img src="/logo_rounded.png" alt="Flood Bell Logo" className={`w-full h-full object-cover ${isLight ? 'mix-blend-normal opacity-90' : 'mix-blend-screen'}`} />
          </motion.div>
          <h1 className={`text-2xl md:text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-b ${isLight ? 'from-blue-900 to-blue-600' : 'from-white to-gray-400'}`}>
            Flood Bell
          </h1>
          <p className="text-brand-accent mt-1 text-center text-[10px] md:text-xs font-bold tracking-widest uppercase">
            Intelligent Evacuation
          </p>
        </motion.div>

        <AnimatePresence mode="wait">
          {error && (
            <motion.div 
              initial={{ opacity: 0, height: 0, mb: 0 }}
              animate={{ opacity: 1, height: 'auto', mb: 20 }}
              exit={{ opacity: 0, height: 0, mb: 0 }}
              className="bg-red-500/10 border border-red-500/50 text-red-500 px-4 py-3 rounded-xl flex items-start text-xs md:text-sm overflow-hidden"
            >
              <AlertCircle size={16} className="mr-2 mt-0.5 flex-shrink-0" />
              <p className="leading-relaxed">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        <form onSubmit={handleEmailAuth} className="space-y-3 md:space-y-4 relative z-10">
          <AnimatePresence>
            {!isLogin && (
              <motion.div 
                initial={{ opacity: 0, height: 0, overflow: 'hidden' }}
                animate={{ opacity: 1, height: 'auto', overflow: 'visible' }}
                exit={{ opacity: 0, height: 0, overflow: 'hidden' }}
                className="grid grid-cols-2 gap-3"
              >
                {/* First Name */}
                <div className="relative group">
                  <User className={`absolute left-3.5 top-1/2 -translate-y-1/2 ${iconClass} group-focus-within:text-brand-primary transition-colors z-10`} size={16} />
                  <input
                    type="text"
                    value={firstName}
                    onChange={(e) => setFirstName(e.target.value)}
                    className={`w-full border rounded-xl py-2.5 pl-10 pr-3 text-sm focus:outline-none focus:border-brand-primary transition-all ${inputBgClass}`}
                    placeholder="First Name"
                    required={!isLogin}
                  />
                </div>
                {/* Last Name */}
                <div className="relative group">
                  <User className={`absolute left-3.5 top-1/2 -translate-y-1/2 ${iconClass} group-focus-within:text-brand-primary transition-colors z-10`} size={16} />
                  <input
                    type="text"
                    value={lastName}
                    onChange={(e) => setLastName(e.target.value)}
                    className={`w-full border rounded-xl py-2.5 pl-10 pr-3 text-sm focus:outline-none focus:border-brand-primary transition-all ${inputBgClass}`}
                    placeholder="Last Name"
                    required={!isLogin}
                  />
                </div>
                {/* Phone */}
                <div className="relative group">
                  <Phone className={`absolute left-3.5 top-1/2 -translate-y-1/2 ${iconClass} group-focus-within:text-brand-primary transition-colors z-10`} size={16} />
                  <input
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    className={`w-full border rounded-xl py-2.5 pl-10 pr-3 text-sm focus:outline-none focus:border-brand-primary transition-all ${inputBgClass}`}
                    placeholder="Phone Number"
                    required={!isLogin}
                  />
                </div>
                {/* DOB */}
                <div className="relative group">
                  <Calendar className={`absolute left-3.5 top-1/2 -translate-y-1/2 ${iconClass} group-focus-within:text-brand-primary transition-colors z-10`} size={16} />
                  <input
                    type="date"
                    value={dob}
                    onChange={(e) => setDob(e.target.value)}
                    className={`w-full border rounded-xl py-2.5 pl-10 pr-3 text-sm focus:outline-none focus:border-brand-primary transition-all ${inputBgClass} ${isDark ? '[color-scheme:dark]' : '[color-scheme:light]'}`}
                    required={!isLogin}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <motion.div variants={itemVariants} className="grid grid-cols-1 gap-3">
            <div className="relative group">
              <Mail className={`absolute left-3.5 top-1/2 -translate-y-1/2 ${iconClass} group-focus-within:text-brand-primary transition-colors z-10`} size={18} />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={`w-full border rounded-xl py-2.5 pl-10 pr-4 text-sm focus:outline-none focus:border-brand-primary transition-all ${inputBgClass}`}
                placeholder="Email Address"
                required
              />
            </div>

            <div className="relative group">
              <Lock className={`absolute left-3.5 top-1/2 -translate-y-1/2 ${iconClass} group-focus-within:text-brand-primary transition-colors z-10`} size={18} />
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={`w-full border rounded-xl py-2.5 pl-10 pr-12 text-sm focus:outline-none focus:border-brand-primary transition-all ${inputBgClass}`}
                placeholder="Password"
                required
              />
              <button 
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className={`absolute right-3.5 top-1/2 -translate-y-1/2 ${iconClass} hover:text-brand-primary transition-colors z-10`}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="pt-2">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-brand-primary to-brand-accent hover:from-blue-500 hover:to-cyan-400 text-white font-semibold py-3 rounded-xl transition-all shadow-[0_0_15px_rgba(59,130,246,0.3)] flex justify-center items-center text-sm md:text-base"
            >
              <AnimatePresence mode="wait">
                {loading ? (
                  <motion.div 
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center space-x-2"
                  >
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                    <span>Authenticating...</span>
                  </motion.div>
                ) : (
                  <motion.span 
                    key="text"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    {isLogin ? 'Sign In Securely' : 'Create Account'}
                  </motion.span>
                )}
              </AnimatePresence>
            </motion.button>
          </motion.div>
        </form>

        <motion.div variants={itemVariants} className="my-5 flex items-center relative z-10">
          <div className="flex-1 border-t border-gray-400/20"></div>
          <span className={`px-3 text-[10px] font-bold tracking-widest ${labelClass}`}>OR</span>
          <div className="flex-1 border-t border-gray-400/20"></div>
        </motion.div>

        <motion.div variants={itemVariants} className="relative z-10">
          <motion.button
            whileHover={{ scale: 1.02, backgroundColor: isLight ? '#f3f4f6' : '#f9fafb' }}
            whileTap={{ scale: 0.98 }}
            onClick={handleGoogleSignIn}
            disabled={loading}
            className={`w-full ${isLight ? 'bg-white border border-gray-300 shadow-sm' : 'bg-white'} text-gray-900 font-semibold py-3 rounded-xl transition-all flex items-center justify-center space-x-2 ${!isLight && 'shadow-lg'} text-sm md:text-base`}
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            <span>Continue with Google</span>
          </motion.button>
        </motion.div>

        <motion.p variants={itemVariants} className={`mt-6 text-center text-xs md:text-sm ${labelClass} relative z-10`}>
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            onClick={() => setIsLogin(!isLogin)} 
            className="text-brand-primary hover:text-brand-accent font-semibold transition-colors"
          >
            {isLogin ? 'Sign up for free' : 'Log in here'}
          </button>
        </motion.p>
      </motion.div>
    </div>
  );
}
