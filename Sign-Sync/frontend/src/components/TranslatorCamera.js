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

  const panelCls = isDarkMode ? "bg-[#0e1a28]/60 border border-white/10" : "bg-gray-200";
  const softPanelCls = panelCls;
  const cardCls = panelCls;
  const textMutedCls = isDarkMode ? "text-gray-300" : "text-gray-700";
  const boxCls = isDarkMode ? "bg-white/5 border border-white/10" : "bg-white";
  const btnBase = "px-3 py-2 border-2 rounded transition-colors";
  const btnNeutral = isDarkMode ? `${btnBase} border-white/20 bg-white/10 hover:bg-white/20`
                                : `${btnBase} border-black bg-gray-300 hover:bg-gray-400`;
  const pillBase = "px-2 py-1 rounded text-sm select-none";
  const pillConnected = "bg-green-600 text-white";
  const pillPaused = "bg-yellow-600 text-white";
  const pillOffline = "bg-red-600 text-white";

  
}
