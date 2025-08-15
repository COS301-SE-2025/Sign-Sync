// import React, { useState } from 'react';

// const WS_URL = 'ws://localhost:8001/api/speech-to-text';
// const CHUNK_SIZE = 4000; // bytes per chunk

// export default function FileStreamer() {
//   const [responses, setResponses] = useState([]);
//   const [elapsedTime, setElapsedTime] = useState(null);

//   const streamFile = async (file) => {
//     const arrayBuffer = await file.arrayBuffer();
//     const totalBytes = arrayBuffer.byteLength;
//     const ws = new WebSocket(WS_URL);

//     ws.binaryType = 'arraybuffer';
//     const start = performance.now();

//     ws.onopen = async () => {
//       let offset = 0;
//       while (offset < totalBytes) {
//         const end = Math.min(offset + CHUNK_SIZE, totalBytes);
//         const chunk = arrayBuffer.slice(offset, end);
//         ws.send(chunk);

//         // wait for server response before sending next chunk
//         await new Promise((resolve) => {
//           ws.onmessage = (event) => {
//             setResponses((prev) => [...prev, event.data]);
//             resolve();
//           };
//         });

//         offset = end;
//       }

//       const finish = performance.now();
//       setElapsedTime(((finish - start) / 1000).toFixed(2));
//       ws.close();
//     };

//     ws.onerror = (err) => console.error('WebSocket error:', err);
//   };

//   const handleFileChange = (e) => {
//     setResponses([]);
//     setElapsedTime(null);
//     const file = e.target.files[0];
//     if (file) streamFile(file);
//   };

//     return (
//         <div style={{ padding: '1rem', fontFamily: 'sans-serif' }}>
//             <h2>Stream .raw Audio File</h2>
//             <input type="file" accept=".raw" onChange={handleFileChange} />

//             {elapsedTime !== null && (
//                 <p>Total time: {elapsedTime} seconds</p>
//             )}

//             <div style={{ marginTop: '1rem' }}>
//                 <strong>Last Response:</strong>
//                 <p>{responses.length > 0 ? responses[responses.length - 1] : 'No response yet.'}</p>
//             </div>
//         </div>
//     );
// }





import React, { useState, useRef } from 'react';

import PreferenceManager from './PreferenceManager';

const SpeechToTextBox = ({onSpeechInput}) => {
  const [text, setText] = useState('');
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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

        try {
          const response = await fetch('http://localhost:8001/api/upload-audio', {
            method: 'POST',
            body: formData,
          });

          const result = await response.json();
          setText(result.text || 'No speech detected');
          onSpeechInput(result.text);
        } catch (error) {
          console.error('Error uploading audio:', error);
          setText('Error transcribing audio');
        }
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setRecording(true);
    } catch (err) {
      console.error('Microphone error:', err);
      alert('Please allow microphone access.');
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
          onClick={toggleRecording} className={`text-2xl font-bold py-2.5 my-2 min-h-[60px] w-1/4 border-2 border-black ${recording ? (isDarkMode ? 'bg-[#36454f] text-white' : 'bg-[#801E20] text-white') : (isDarkMode ? 'bg-[#801E20] text-black' : 'bg-[#36454f] text-white')}`}>
          {recording ? 'Stop' : 'Speak 🎙️'}
        </button>
      </div>
    </div>
  );
};

export default SpeechToTextBox;
