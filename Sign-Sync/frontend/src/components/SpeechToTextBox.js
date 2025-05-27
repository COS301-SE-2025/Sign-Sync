import React, { useRef, useState } from 'react';

const SpeechToTextBox = () => {
    const [text, setText] = useState('');
    const [recording, setRecording] = useState(false);
    const websocketRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const streamRef = useRef(null); // To stop audio tracks

    const startRecording = async () => {
        try {
            // const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000, // might be ignored by some browsers but still useful
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            });
            streamRef.current = stream;
            const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            const ws = new WebSocket("ws://localhost:8001/api/speech-to-text");
            // websocketRef.current = new WebSocket("ws://127.0.0.1:8000/api/speech-to-text");

            ws.onopen = () => {
                console.log("WebSocket connected");

                ws.onmessage = (event) => {
                    if (event.data) {
                        setText(prev => event.data); // Overwrite text each message
                    }
                };

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                        e.data.arrayBuffer().then(buffer => {
                            ws.send(buffer);
                        });
                    }
                };

                // mediaRecorder.ondataavailable = async (e) => {
                //     if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                //         const arrayBuffer = await e.data.arrayBuffer();

                //         const audioCtx = new AudioContext({ sampleRate: 16000 }); // Force 16kHz
                //         const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

                //         const monoBuffer = audioCtx.createBuffer(
                //             1,
                //             audioBuffer.length,
                //             audioBuffer.sampleRate
                //         );

                //         // Convert to mono by taking the first channel only
                //         const input = audioBuffer.getChannelData(0);
                //         const output = monoBuffer.getChannelData(0);
                //         for (let i = 0; i < input.length; i++) {
                //             output[i] = input[i];
                //         }

                //         // Convert to 16-bit PCM
                //         const pcmBuffer = new ArrayBuffer(output.length * 2);
                //         const pcmView = new DataView(pcmBuffer);
                //         for (let i = 0; i < output.length; i++) {
                //             const s = Math.max(-1, Math.min(1, output[i]));
                //             pcmView.setInt16(i * 2, s * 0x7FFF, true); // little endian
                //         }

                //         // websocketRef.current.send(pcmBuffer);
                //         ws.send(pcmBuffer);
                //     }
                // };


                mediaRecorder.start(250); // Chunk every 250ms
                mediaRecorderRef.current = mediaRecorder;
                websocketRef.current = ws;
                setRecording(true);
            };

            ws.onerror = (err) => {
                console.error("WebSocket error:", err);
            };

        } catch (err) {
            console.error('Error accessing microphone:', err);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            mediaRecorderRef.current = null;
        }

        if (websocketRef.current) {
            websocketRef.current.close();
            websocketRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        setRecording(false);
    };

    // const stopRecording = () => {
    //     if (mediaRecorderRef.current) {
    //         mediaRecorderRef.current.stop();
    //     }

    //     // 👇 Wait 1 second before closing WebSocket
    //     setTimeout(() => {
    //         if (websocketRef.current) {
    //             websocketRef.current.close();
    //         }
    //     }, 1000);

    //     setRecording(false);
    // };

    const toggleRecording = () => {
        recording ? stopRecording() : startRecording();
    };

    return (
        <div className="mt-4">
            <div className="flex items-center gap-2">
                <input
                    type="text"
                    value={text}
                    readOnly
                    className="p-2 border rounded w-2/3"
                    placeholder="Speech will appear here..."
                />
                <button
                    onClick={toggleRecording}
                    className={`p-2 rounded ${recording ? 'bg-red-500' : 'bg-green-500'} text-white`}
                >
                    {recording ? 'Stop' : 'Speak 🎙️'}
                </button>
            </div>
        </div>
    );
};

export default SpeechToTextBox;


// import React, { useRef, useState } from 'react';

// const SpeechToTextBox = () => {
//     const [text, setText] = useState('');
//     const [recording, setRecording] = useState(false);
//     const mediaRecorderRef = useRef(null);
//     const audioChunksRef = useRef([]);

//     const startRecording = async () => {
//         const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//         const mediaRecorder = new MediaRecorder(stream);

//         mediaRecorder.ondataavailable = event => {
//             if (event.data.size > 0) {
//                 audioChunksRef.current.push(event.data);
//             }
//         };

//         mediaRecorder.onstop = async () => {
//             const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
//             const formData = new FormData();
//             formData.append('file', audioBlob, 'recording.webm');

//             const response = await fetch('http://localhost:8001/api/upload-audio', {
//                 method: 'POST',
//                 body: formData
//             });

//             const data = await response.json();
//             setText(data.text || 'No text detected');
//             audioChunksRef.current = [];
//         };

//         mediaRecorderRef.current = mediaRecorder;
//         mediaRecorder.start();
//         setRecording(true);
//     };

//     const stopRecording = () => {
//         mediaRecorderRef.current?.stop();
//         setRecording(false);
//     };

//     const toggleRecording = () => {
//         recording ? stopRecording() : startRecording();
//     };

//     return (
//         <div className="mt-4">
//             <div className="flex items-center gap-2">
//                 <input
//                     type="text"
//                     value={text}
//                     readOnly
//                     className="p-2 border rounded w-2/3"
//                     placeholder="Speech will appear here..."
//                 />
//                 <button
//                     onClick={toggleRecording}
//                     className={`p-2 rounded ${recording ? 'bg-red-500' : 'bg-green-500'} text-white`}
//                 >
//                     {recording ? 'Stop' : 'Speak 🎙️'}
//                 </button>
//             </div>
//         </div>
//     );
// };

// export default SpeechToTextBox;
