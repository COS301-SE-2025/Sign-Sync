import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import hand_landmarker_task from "../assets/hand_landmarker.task";

import SoundOnIcon from "../assets/SoundOn.png";
import SoundOffIcon from "../assets/SoundOff.png";
import conversationIcon from "../assets/conversation.png";

const Camera = () => {
    const videoRef = useRef(null);
    const [handPresence, setHandPresence] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [soundOn, setSoundOn] = useState(null);

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
                scanWebcam();
            } catch (error) {
                console.error(error);
            }
        };

        const scanWebcam = () => {
            const loop = setInterval(() => {
                if (videoRef.current && handLandmarker) {
                    const detections = handLandmarker.detectForVideo(videoRef.current, performance.now());
                    if (detections.landmarks && detections.landmarks.length > 0) {
                        makePrediction(detections.landmarks[0]);
                    }else{
                        setPrediction("No hand detected");
                    }
                }
            },500);
            return () => clearInterval(loop);
        };

        const makePrediction = async (keypointArray) => {
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
        };
    }, []);

    const changeSound = () => {
        setSoundOn(sound => !sound);
    }

    return (
        <div>
            <div className="bg-gray-200 p-2 rounded-lg mb-2">
                <video ref={videoRef} autoPlay playsInline width="640" height="400" />
            </div>
            <div class="flex items-center border bg-gray-200 rounded-lg px-4 py-2 ">
                <button className="bg-gray-300 p-3.5 border-2 border-black"><img src={conversationIcon} className="w-8 h-8" alt={"Conversation"}/></button>
                <h1 className="text-center w-3/4 text-4xl font-bold border-2 border-black bg-gray-300 py-2.5 my-2 justify-center flex flex-grow min-h-[60px] ">{prediction}</h1>
                <button onClick={changeSound} className="bg-gray-300 p-3.5 border-2 border-black"><img src={soundOn? SoundOnIcon : SoundOffIcon} className="w-8 h-8" alt={"Speaker"}/></button>
            </div>
        </div>
    );
};

export default Camera;
