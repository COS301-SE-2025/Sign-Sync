import React, { useState, useRef, useEffect } from 'react';
import { toast } from "react-toastify";

import PreferenceManager from './PreferenceManager';

const SpeechToTextBox = ({onSpeechInput}) => {
  const [text, setText] = useState('');
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const [isUploading, setIsUploading] = useState(false);              
  const [lockoutUntil, setLockoutUntil] = useState(null);             
  const [remainingSecs, setRemainingSecs] = useState(0);              
  const countdownTimerRef = useRef(null);                             
  const streamRef = useRef(null);                                     

  const isLockedOut = !!lockoutUntil && Date.now() < lockoutUntil;    

  useEffect(() => {                                                   
    return () => {                                                    
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current); 
      // stop any open stream on unmount                                
      if (streamRef.current) {                                        
        try { streamRef.current.getTracks().forEach(t => t.stop()); } catch {} 
        streamRef.current = null;                                     
      }                                                               
    };                                                                
  }, []);                                                             

  const startLockout = (secs) => {                                    
    const until = Date.now() + secs * 1000;                           
    setLockoutUntil(until);                                           
    setRemainingSecs(secs);                                           
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current); 
    countdownTimerRef.current = setInterval(() => {                   
      const rem = Math.max(0, Math.ceil((until - Date.now()) / 1000)); 
      setRemainingSecs(rem);                                          
      if (rem <= 0) {                                                 
        clearInterval(countdownTimerRef.current);                     
        countdownTimerRef.current = null;                             
        setLockoutUntil(null);                                        
      }                                                               
    }, 1000);                                                         
  };                                                                  

  const startRecording = async () => {

    if (isUploading) {                                                
      toast.info("Please wait for the current upload to finish.");    
      return;                                                         
    }                                                                 
    if (isLockedOut) {                                                
      toast.error(`Too many requests. Try again in ${remainingSecs}s.`); 
      return;                                                         
    }                                                                 

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;                                     
      const mediaRecorder = new MediaRecorder(stream);

      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');

        setIsUploading(true);                                         

        try {
          const response = await fetch('http://localhost:8007/api/speech/api/upload-audio', {
          //const response = await fetch('https://apigateway-evbsd4dmhbbyhwch.southafricanorth-01.azurewebsites.net/api/speech/api/upload-audio', { //deployment version
            method: 'POST',
            body: formData,
          });

          // const result = await response.json();
          // setText(result.text || 'No speech detected');
          // onSpeechInput(result.text);
          if (response.ok) {                                          
            const result = await response.json();                     
            const out = result?.text || 'No speech detected';         
            setText(out);                                             
            onSpeechInput && onSpeechInput(result?.text || "");       
          } 
          else {                                                    
            // Safely parse JSON; may be empty on some errors          
            let msg = "";                                             
            try { msg = (await response.json())?.message || ""; } catch {} 
            if (response.status === 429) {                            
              const ra = response.headers.get("Retry-After");         
              const secs = ra && /^\d+$/.test(ra) ? parseInt(ra, 10) : 30; 
              startLockout(secs);                                     
              toast.error(msg || `Too many requests. Try again in ${secs}s.`); 
              // Don’t overwrite existing text with undefined           
            } 
            else {                                                  
              setText('Error transcribing audio');                    
              toast.error(msg || 'Error transcribing audio');         
            }                                                         
          }                                                           
        } 
        catch (error) 
        {
          console.error('Error uploading audio:', error);
          setText('Error transcribing audio');
          toast.error('Network error during audio upload.');          
        }
        finally {
          setIsUploading(false);                                      
          // release the mic after upload                              
          if (streamRef.current) {                                    
            try { streamRef.current.getTracks().forEach(t => t.stop()); } catch {} 
            streamRef.current = null;                                 
          }                                                           
        }
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setRecording(true);
    } catch (err) {
      console.error('Microphone error:', err);
      toast.error('Please allow microphone access.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const toggleRecording = () => {
    recording ? stopRecording() : startRecording();
  };

  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <div className=" items-center mx-auto">
      <div className="flex items-stretch">
        <input
          type="text"
          value={text}
          readOnly
          className={`text-center w-3/4 text-4xl font-bold border-2 border-black py-2.5 my-2 flex-grow min-h-[60px] ${isDarkMode ? "bg-gray-700 text-white" : "bg-gray-300 text-black"}`}
          placeholder="Speech will appear here..."
        />

        <button
          onClick={toggleRecording}
          disabled={isUploading || isLockedOut}
          className={`text-2xl font-bold py-2.5 my-2 min-h-[60px] w-1/4 border-2 border-black ${recording ? 'bg-indigo-700 text-white hover:bg-indigo-800' : (isUploading || isLockedOut) ? 'bg-gray-400 text-white cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`}>
          {recording ? 'Stop' : isUploading ? 'Uploading…' : isLockedOut ? `Cooldown ${remainingSecs}s` : 'Speak 🎙️'}
        </button>
      </div>
    </div>
  );
};

export default SpeechToTextBox;
