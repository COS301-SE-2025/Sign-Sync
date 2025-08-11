import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import hand_landmarker_task from "../assets/hand_landmarker.task";

import SoundOnIcon from "../assets/SoundOn.png";
import SoundOffIcon from "../assets/SoundOff.png";
import gestureIcon from "../assets/Gestures.png";
import letterIcon from "../assets/Letters.png";
import {temp} from "three/src/Three.TSL";

const Camera = ( {defaultGestureMode = true, gestureModeFixed = false, onPrediction} ) => {
    const videoRef = useRef(null);
    const [handPresence, setHandPresence] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [soundOn, setSoundOn] = useState(null);
    // const [gestureMode, setGestureMode] = useState(true); /////////////////////////////////////////////
    const [gestureMode, setGestureMode] = useState(defaultGestureMode);

    const speakText = (text) => {
        if (!("speechSynthesis" in window)) 
        {
            console.warn("Text-to-Speech not supported in this browser.");
            return;
        }

        if (!text || text === "No hand detected") return;

        window.speechSynthesis.cancel();

        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = "en-US";
        utter.rate = 1;
        utter.pitch = 1;

        utter.onstart = () => setSoundOn(true);
        utter.onend = () => setSoundOn(false);
        utter.onerror = () => setSoundOn(false);

        window.speechSynthesis.speak(utter);
    };

    useEffect(() => {
        let handLandmarker;
        let animationFrameId;

        //https://dev.to/kiyo/integrating-mediapipetasks-vision-for-hand-landmark-detection-in-react-2lbg
        const initializeHandDetection = async () => {
            try {
                const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",);
                handLandmarker = await HandLandmarker.createFromOptions(
                    vision, {
                        baseOptions: { modelAssetPath: hand_landmarker_task },numHands: 2, runningMode: "video"}
                );
                animationFrameId = scanWebcam();
            } catch (error) {
                console.error(error);
            }
        };

        const scanWebcam = () => {
            let arrayLandmarks = [];
            let start = Date.now();
            console.log(start);
            const loop = setInterval(() => {
                if (videoRef.current && handLandmarker) {
                    if(gestureMode) {
                        const detections = handLandmarker.detectForVideo(videoRef.current, performance.now());
                        if (detections.landmarks && detections.landmarks.length > 0) {
                            const tempArr = [];
                            for (let i = 0; i < detections.landmarks.length; i++) {
                                const temp = detections.landmarks[i].flatMap(({ x, y, z }) => [x, y, z]);
                                tempArr.push(...temp);
                            }
                            if (tempArr.length < 126) {
                                const blankArr = Array(126-tempArr.length).fill(0.0);
                                tempArr.push(...blankArr);
                            }
                            if(arrayLandmarks.length === 0) {
                                arrayLandmarks = [tempArr];
                            }else{
                                arrayLandmarks = [...arrayLandmarks,tempArr];
                            }
                            if(arrayLandmarks.length === 50) {
                                makePredictionGesture(arrayLandmarks);
                                console.log(Date.now()-start);
                                start = Date.now();
                                arrayLandmarks = [];
                            }
                        }
                    }else{
                        const detections = handLandmarker.detectForVideo(videoRef.current, performance.now());
                        if (detections.landmarks && detections.landmarks.length > 0) {
                            makePredictionLetters(detections.landmarks[0]);
                        }else{
                            setPrediction("No hand detected");
                        }
                    }
                }
            },gestureMode? 40 : 500);
            return loop;
        };
        const makePredictionGesture = async (arrayLandmarks) => {

            try {
                const request = await fetch("http://localhost:8003/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({sequence:arrayLandmarks})
                });

                const response = await request.json();
                setPrediction(response.gloss);
            } catch (err) {
                console.error("Failed to fetch prediction:", err);
            }
        };
        const makePredictionLetters = async (keypointArray) => {
            const landmarkJSON = {
                keypoints: keypointArray.map((coordinates) => ({
                    x: coordinates.x,
                    y: coordinates.y,
                    z: coordinates.z
                }))
            };

            try {
                const request = await fetch("http://localhost:8000/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(landmarkJSON)
                });

                const response = await request.json();
                setPrediction(response.prediction);
                if(onPrediction) {
                    onPrediction(response.prediction);
                }
            } catch (err) {
                console.error("Failed to fetch prediction:", err);
            }
        };

        const startWebcam = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoRef.current.srcObject = stream;
                videoRef.current.style.transform = "scaleX(-1)";
                await initializeHandDetection();
            } catch (error) {
                console.error(error);
            }
        };

        startWebcam();

        return () => {
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
            if (handLandmarker) {
                handLandmarker.close();
            }

            if (animationFrameId) {
                clearInterval(animationFrameId);
            }
        };
    }, [gestureMode]);

    const changeSound = () => {
        setSoundOn(sound => !sound);
    }
    const changeGestureMode = () => {
        setGestureMode(gestureMode => !gestureMode);
    }
    return (
        <div>
            <div className="bg-gray-200 p-2 rounded-lg mb-2">
                <video ref={videoRef} autoPlay playsInline width="640" height="400" />
            </div>
            <div className="flex items-center border bg-gray-200 rounded-lg px-4 py-2 ">
                
                {!gestureModeFixed && (<button onClick={changeGestureMode} className="bg-gray-300 p-3.5 border-2 border-black"><img src={gestureMode? gestureIcon : letterIcon} className="w-8 h-8" alt={"Conversation"}/></button> )}
                
                <h1 className="text-center w-3/4 text-4xl font-bold border-2 border-black bg-gray-300 py-2.5 my-2 justify-center flex flex-grow min-h-[60px] ">{prediction}</h1>
                <button onClick={changeSound} className="bg-gray-300 p-3.5 border-2 border-black"><img src={soundOn? SoundOnIcon : SoundOffIcon} className="w-8 h-8" alt={"Speaker"}/></button>
            </div>
        </div>
    );
};

export default Camera;
