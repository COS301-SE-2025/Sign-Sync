import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import hand_landmarker_task from "../assets/hand_landmarker.task";

const Camera = () => {
    const videoRef = useRef(null);
    const [handPresence, setHandPresence] = useState(null);
    const [prediction, setPrediction] = useState(null);

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

    return (
        <div className={"flex flex-col items-center justify-center min-h-screen"}>
            <video ref={videoRef} autoPlay playsInline width="640" height="480" />
            <div>
                <h1 className="text-4xl font-bold">{prediction}</h1>
            </div>
        </div>
    );
};

export default Camera;
