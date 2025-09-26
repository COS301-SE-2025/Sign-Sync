import React, { useEffect, useRef, useState } from "react";
// import { FilesetResolver, PoseLandmarker, HandLandmarker } from "@mediapipe/tasks-vision";

import PreferenceManager from "./PreferenceManager";
import CameraFeed from "./CameraFeed";
import { useSignStream } from "./useSignStream";

import SoundOnIcon from "../assets/SoundOn.png";
import SoundOffIcon from "../assets/SoundOff.png";
import gestureIcon from "../assets/Gestures.png";
import letterIcon from "../assets/Letters.png";

export default function TranslatorCamera({ onPrediction }) {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
  const [gestureMode, setGestureMode] = React.useState(true);

  const {
    videoRef, connected, status, paused,
    headline, topK, stable, sentence, setSentence,
    soundOn,
    start, stop, pause, resume, undo, clear, toEnglish, toggleSpeak,
  } = useSignStream({ mode: gestureMode ? "words" : "letters", onPrediction, autoStart: true });

  const card = isDarkMode
    ? "bg-white/5 border border-white/10"
    : "bg-white border border-black/10";
  const soft = isDarkMode
    ? "bg-white/5 border border-white/10"
    : "bg-gray-100 border border-black/10";
  const btnBase = "px-3 py-2 rounded-lg border text-sm font-medium transition";
  const btnNeutral = isDarkMode
    ? `${btnBase} border-white/15 bg-white/10 hover:bg-white/20`
    : `${btnBase} border-black/10 bg-gray-200 hover:bg-gray-300`;

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-stretch">
      {/* ROW 1 — LEFT: CAMERA */}
      <div className="md:col-span-8 md:row-start-1">
        <div className="relative rounded-2xl overflow-hidden shadow-lg">
          <CameraFeed videoRef={videoRef} />

          
    </div>
  );
}
