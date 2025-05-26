import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import hand_landmarker_task from "../assets/hand_landmarker.task";

const Camera = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [handPresence, setHandPresence] = useState(null);

    useEffect(() => {
        let handLandmarker;

        //https://medium.com/@mamikonyanmichael/what-is-media-pipe-and-how-to-use-it-in-react-53ff418e5a68
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

        const getLandmarks = (array) => {
            array.forEach(landmarks => {
                landmarks.forEach(landmark => {
                    console.log(landmark);
                });
            });
        };

        const scanWebcam = () => {
            if (videoRef.current && videoRef.current.readyState >= 2) {
                const detections = handLandmarker.detectForVideo(videoRef.current, performance.now());
                setHandPresence(detections.handednesses.length > 0);

                // Assuming detections.landmarks is an array of landmark objects
                if (detections.landmarks) {
                    getLandmarks(detections.landmarks);
                }
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
        <>
            <div style={{ position: "relative"}}>
                <video ref={videoRef} autoPlay playsInline ></video>
            </div>
        </>
    );
};

export default Camera;