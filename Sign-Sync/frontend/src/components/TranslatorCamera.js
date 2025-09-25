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

  return (
    <div>
      <div className={`${panelCls} p-2 rounded-lg mb-2`}>
        <CameraFeed videoRef={videoRef} />
      </div>

      <div className={`flex items-center gap-3 rounded-lg px-4 py-3 ${softPanelCls}`}>
        <span
          onClick={() => { if (gestureMode && connected) { paused ? resume() : pause(); } }}
          className={`${pillBase} cursor-${gestureMode && connected ? "pointer" : "default"} ${connected ? (gestureMode ? (paused ? pillPaused : pillConnected) : pillConnected) : pillOffline}`}
          title={gestureMode ? (connected ? (paused ? "Click to resume" : "Click to pause") : "Not connected") : "Pause only in Words mode"}
        >
          {connected ? (gestureMode ? (paused ? "Paused" : `Connected (${status})`) : "Letters (Polling)") : "Offline"}
        </span>

        <button onClick={() => setGestureMode(v => !v)} className={btnNeutral}>
          <img src={gestureMode ? gestureIcon : letterIcon} className="w-8 h-8" alt="Mode" />
        </button>

        <button onClick={undo} className={btnNeutral} disabled={!gestureMode}>Undo</button>
        <button onClick={clear} className={btnNeutral} disabled={!gestureMode}>Clear</button>

        <div className="flex-1" />

        <button onClick={toggleSpeak} className={btnNeutral}>
          <img src={SoundOnIcon} className={`w-8 h-8 ${soundOn ? "" : "opacity-40"}`} alt="Speaker" />
        </button>
      </div>

      <div className={`mt-3 rounded-lg p-3 ${cardCls}`}>
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold">
            {gestureMode ? (headline ? `${headline}${stable ? " ✅" : ""}` : "—") : (headline || "—")}
          </h2>
          {gestureMode && (
            <div className={`text-sm ${textMutedCls}`}>
              {topK.map((t, i) => <span key={i} className="mr-3">{i + 1}. {t.label} {(t.p * 100).toFixed(1)}%</span>)}
            </div>
          )}
        </div>
      </div>

      <div className="mt-3">
        <h3 className="text-lg font-semibold mb-1">Sentence</h3>
        <div className="flex rounded min-h-[60px]">
          <div className={`p-3 text-2xl w-4/5 ${boxCls}`}>{gestureMode ? (sentence || " ") : " "}</div>
          <div className="w-1/5 flex items-center justify-center">
            <button onClick={() => toEnglish(sentence)} className="px-3 py-2 bg-blue-500 text-white rounded">To English</button>
          </div>
        </div>
      </div>
    </div>
  );
}
