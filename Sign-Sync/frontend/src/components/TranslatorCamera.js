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

          {/* Connected pill */}
          <button
            onClick={() => { if (gestureMode && connected) { paused ? resume() : pause(); } }}
            className={`absolute top-3 left-3 px-3 py-1.5 rounded-lg text-xs font-semibold shadow
            ${connected ? (gestureMode ? (paused ? "bg-yellow-600 text-white" : "bg-green-600 text-white")
                : "bg-green-600 text-white")
                : "bg-red-600 text-white"}`}
            title={gestureMode ? (connected ? (paused ? "Click to resume" : "Click to pause") : "Not connected") : "Pause only in Words mode"}
          >
            {connected ? (gestureMode ? (paused ? "Paused" : `Connected (${status})`) : "Connected (Alphabet)") : "Offline"}
          </button>

          {/* Alphabet toggle */}
          <button
            onClick={() => setGestureMode(v => !v)}
            className="absolute top-3 right-3 px-3 py-1.5 rounded-lg border bg-white/90 text-black"
          >
            {gestureMode ? "Alphabet" : "Words"}
          </button>
        </div>
      </div>

      {/* ROW 1 — RIGHT: PREDICTED CARD (match camera height) */}
      <aside className="md:col-span-4 md:row-start-1 h-full self-stretch">
        <div className={`${card} rounded-2xl p-5 h-full flex flex-col text-base md:text-lg`}>
          <h3 className="text-2xl md:text-3xl font-extrabold tracking-tight mb-4">
            Predicted Word
          </h3>

          <div className="text-4xl md:text-5xl font-black leading-none mb-4 break-words">
            {gestureMode ? (headline ? `${headline}${stable ? " ✅" : ""}` : "—") : (headline || "—")}
          </div>

          <ol className="space-y-3">
            {topK.slice(0, 3).map((t, i) => (
              <li key={i} className="flex items-center justify-between">
                <span className="font-semibold text-lg md:text-xl">
                  {i + 1}. {t.label}
                </span>
                <span className="opacity-80 text-sm md:text-base">
                  {(t.p * 100).toFixed(1)}%
                </span>
              </li>
            ))}
          </ol>

          
    </div>
  );
}
