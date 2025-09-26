import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker, HandLandmarker } from "@mediapipe/tasks-vision";

import pose_task from "../assets/pose_landmarker_full.task";
import hand_task from "../assets/hand_landmarker.task";

import SoundOnIcon from "../assets/SoundOn.png";
import SoundOffIcon from "../assets/SoundOff.png";
import gestureIcon from "../assets/Gestures.png";
import letterIcon from "../assets/Letters.png";
import PreferenceManager from "../components/PreferenceManager"; // <-- add this

// --- endpoints ---
const WORDS_API_BASE = "http://localhost:8007/api/stt"; // words model (WS + REST)
const LETTERS_API_BASE = "http://localhost:8007/api/alphabet"; // letters model (REST)
const GRAMMAR_API_BASE = "http://localhost:8007/api/word"; // grammar model (REST)

//deployment version:
// const WORDS_API_BASE = "https://apigateway-evbsd4dmhbbyhwch.southafricanorth-01.azurewebsites.net/api/stt"; // words model (WS + REST)
// const LETTERS_API_BASE = "https://apigateway-evbsd4dmhbbyhwch.southafricanorth-01.azurewebsites.net/api/alphabet"; // letters model (REST)
// const GRAMMAR_API_BASE = "https://apigateway-evbsd4dmhbbyhwch.southafricanorth-01.azurewebsites.net/api/word"; // grammar model (REST)

const SEND_INTERVAL_MS = 80;   // words streaming cadence
const LETTERS_INTERVAL_MS = 500; // letters polling cadence

// helpers
const collapseConsecutive = (text) => {
  const toks = (text || "").trim().split(/\s+/).filter(Boolean);
  if (!toks.length) return "";
  const out = [];
  let prev = null;
  for (const t of toks) { if (t !== prev) out.push(t); prev = t; }
  return out.join(" ") + " ";
};
const lastWord = (text) => {
  const toks = (text || "").trim().split(/\s+/).filter(Boolean);
  return toks.length ? toks[toks.length - 1] : "";
};

const TranslatorCamera = ({ onPrediction }) => {
  // ---- THEME ----
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  // neutrals
  const panelCls = isDarkMode ? "bg-[#0e1a28]/60 border border-white/10" : "bg-gray-200";
  const softPanelCls = isDarkMode ? "bg-[#0e1a28]/60 border border-white/10" : "bg-gray-200";
  const cardCls = isDarkMode ? "bg-[#0e1a28]/60 border border-white/10" : "bg-gray-200";
  const textMutedCls = isDarkMode ? "text-gray-300" : "text-gray-700";
  const boxCls = isDarkMode ? "bg-white/5 border border-white/10" : "bg-white";

  // buttons
  const btnBase = "px-3 py-2 border-2 rounded transition-colors";
  const btnNeutral = isDarkMode
    ? `${btnBase} border-white/20 bg-white/10 hover:bg-white/20`
    : `${btnBase} border-black bg-gray-300 hover:bg-gray-400`;

  // status pill colors stay semantic; just tweak contrast in dark
  const pillBase = "px-2 py-1 rounded text-sm select-none";
  const pillConnected = isDarkMode ? "bg-green-600 text-white" : "bg-green-600 text-white";
  const pillPaused = isDarkMode ? "bg-yellow-600 text-white" : "bg-yellow-600 text-white";
  const pillOffline = isDarkMode ? "bg-red-600 text-white" : "bg-red-600 text-white";

  const videoFrameCls = `${panelCls} p-2 rounded-lg mb-2`;

  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const poseRef = useRef(null);
  const handRef = useRef(null);
  const loopTimerRef = useRef(null);
  const sessionIdRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [soundOn, setSoundOn] = useState(false);

  const [topK, setTopK] = useState([]); // [{label,p}]
  const [stable, setStable] = useState(false);
  const [sentence, setSentence] = useState("");
  const [headline, setHeadline] = useState("");

  const lastCommittedRef = useRef("");

  const [gestureMode, setGestureMode] = useState(true);   // true = WORDS, false = LETTERS
  const [gestureModeFixed] = useState(false);
  const [paused, setPaused] = useState(false);

  const startSendLoopWords = () => {
    if (loopTimerRef.current) return;
    loopTimerRef.current = setInterval(() => tickSendWords(), SEND_INTERVAL_MS);
  };

  const stopSendLoop = () => {
    if (loopTimerRef.current) {
      clearInterval(loopTimerRef.current);
      loopTimerRef.current = null;
    }
  };

  const pauseWords = () => {
    if (!gestureMode) return;           // only valid in words mode
    stopSendLoop();
    setPaused(true);
    setStatus("Paused");
  };

  const resumeWords = () => {
    if (!gestureMode) return;
    setPaused(false);
    setStatus("Predicting");
    startSendLoopWords();
  };

  const speakText = (text) => {
    if (!("speechSynthesis" in window) || !text) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = "en-US";
    utt.rate = 1.0;
    utt.pitch = 1.0;
    utt.onstart = () => setSoundOn(true);
    utt.onend = () => setSoundOn(false);
    utt.onerror = () => setSoundOn(false);
    window.speechSynthesis.speak(utt);
  };

  const toggleSpeak = () => {
    const text = (sentence || "").replace(/\s+/g, " ").trim();
    if (!text) {
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        setSoundOn(false);
      }
      return;
    }
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      setSoundOn(false);
    } else {
      speakText(text);
    }
  };

  // ---------- WORDS MODE (WS to 8004) ----------
  const startWordsSession = async () => {
    const resp = await fetch(`${WORDS_API_BASE}/v1/session/start`, { method: "POST" });
    if (!resp.ok) throw new Error("Failed to start session");
    const meta = await resp.json();
    sessionIdRef.current = meta.session_id;

    const ws = new WebSocket(`ws://localhost:8007/api/stt/v1/stream/${meta.session_id}`);
    //const ws = new WebSocket(`wss://apigateway-evbsd4dmhbbyhwch.southafricanorth-01.azurewebsites.net/api/stt/v1/stream/${meta.session_id}`); //deployment version

    wsRef.current = ws;

    ws.onopen = () => { setConnected(true); setStatus("Idle"); };
    ws.onclose = () => { setConnected(false); setStatus("Idle"); };
    ws.onerror = () => { try { ws.close(); } catch { } };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "prediction") {
          if (msg.idle) setStatus("Idle");
          else if (msg.filling) setStatus("Filling");
          else setStatus("Predicting");

          setTopK(msg.topk || []);
          setStable(!!msg.stable);
          if (msg.topk?.length) {
            setHeadline(msg.topk[0].label);
            onPrediction && onPrediction(msg.topk[0].label, msg.topk);
          } else {
            setHeadline("");
          }
        } else if (msg.type === "word_event") {
          const w = (msg.label || "").trim();
          if (w && w !== lastCommittedRef.current) lastCommittedRef.current = w;
        } else if (msg.type === "sentence") {
          const cleaned = collapseConsecutive(msg.text || "");
          setSentence(cleaned);
          lastCommittedRef.current = lastWord(cleaned);
        }
      } catch { }
    };

    setPaused(false);
    startSendLoopWords();
  };

  const tickSendWords = () => {
    const videoEl = videoRef.current;
    const pose = poseRef.current;
    const hand = handRef.current;
    const ws = wsRef.current;
    if (!videoEl || !pose || !hand || !ws || ws.readyState !== WebSocket.OPEN) return;

    const ts = performance.now();

    const pRes = pose.detectForVideo(videoEl, ts);
    let pose33 = null;
    if (pRes?.landmarks?.length) {
      const first = pRes.landmarks[0];
      if (first && first.length === 33) pose33 = first.map((lm) => [lm.x, lm.y, lm.z ?? 0, lm.visibility ?? 0]);
    }
    if (!pose33) return;

    const hRes = hand.detectForVideo(videoEl, ts);
    let left21 = Array.from({ length: 21 }, () => [0, 0, 0]);
    let right21 = Array.from({ length: 21 }, () => [0, 0, 0]);

    if (hRes?.landmarks?.length) {
      const LM = hRes.landmarks;
      const H = hRes.handednesses;
      if (H && H.length === LM.length) {
        for (let i = 0; i < LM.length; i++) {
          const side = (H[i][0]?.categoryName || "").toLowerCase();
          const pts = LM[i].map((lm) => [lm.x, lm.y, lm.z ?? 0]);
          if (side === "left") left21 = pts;
          else if (side === "right") right21 = pts;
        }
      } else {
        const pts0 = LM[0]?.map((lm) => [lm.x, lm.y, lm.z ?? 0]);
        if (pts0) left21 = pts0;
        const pts1 = LM[1]?.map((lm) => [lm.x, lm.y, lm.z ?? 0]);
        if (pts1) right21 = pts1;
      }
    }

    const payload = { t: Date.now(), pose33, left21, right21 };
    try { ws.send(JSON.stringify(payload)); } catch { }
  };

  const stopWordsSession = async () => {
    stopSendLoop();
    setPaused(false);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try { wsRef.current.close(); } catch { }
    }
    if (sessionIdRef.current) {
      try {
        await fetch(`${WORDS_API_BASE}/v1/session/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionIdRef.current }),
        });
      } catch { }
      sessionIdRef.current = null;
    }
    setConnected(false);
    setStatus("Idle");
  };

  // ---------- LETTERS MODE (HTTP to 8000) ----------
  const startLettersLoop = () => {
    loopTimerRef.current = setInterval(() => tickSendLetters(), LETTERS_INTERVAL_MS);
    setConnected(true);
    setStatus("Predicting");
  };

  const tickSendLetters = async () => {
    const videoEl = videoRef.current;
    const hand = handRef.current;
    if (!videoEl || !hand) return;

    const ts = performance.now();
    const hRes = hand.detectForVideo(videoEl, ts);
    const lm = hRes?.landmarks?.[0];
    if (!lm || !lm.length) {
      setHeadline("");
      return;
    }

    const keypoints = lm.map((pt) => ({ x: pt.x, y: pt.y, z: pt.z ?? 0 }));
    try {
      const resp = await fetch(`${LETTERS_API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keypoints }),
      });
      if (!resp.ok) return;
      const data = await resp.json(); // expect { prediction: "A" }
      const pred = (data.prediction || "").toString();
      setHeadline(pred);
      setTopK([]);
      setStable(true);
      onPrediction && onPrediction(pred, []);
    } catch (e) {
      // swallow errors in loop
    }
  };

  const toEnglish = async (text) => {
    if (!text) return "";
    try {
      const resp = await fetch(`${GRAMMAR_API_BASE}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setSentence(data.translation || sentence);
      }
    } catch (e) {
      console.error("Failed to convert to English:", e);
      return "";
    }
  }

  const stopLettersLoop = () => {
    if (loopTimerRef.current) { clearInterval(loopTimerRef.current); loopTimerRef.current = null; }
    setConnected(false);
    setStatus("Idle");
  };

  // ---------- shared setup / teardown ----------
  useEffect(() => {
    let stream = null;

    const initMediaPipe = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      poseRef.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: pose_task },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      handRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: hand_task },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    };

    const start = async () => {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 400 } });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      videoRef.current.style.transform = "scaleX(-1)";
      await videoRef.current.play().catch(() => { });
      await initMediaPipe();

      if (gestureMode) {
        await startWordsSession();
      } else {
        startLettersLoop();
      }
    };

    start().catch((e) => console.error("Camera init failed:", e));

    return () => {
      stopLettersLoop();
      stopWordsSession().catch(() => { });
      if (poseRef.current) { poseRef.current.close(); poseRef.current = null; }
      if (handRef.current) { handRef.current.close(); handRef.current = null; }
      if (videoRef.current && stream) {
        stream.getTracks().forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      }
    };
  }, [gestureMode, onPrediction]);

  // ---------- UI actions ----------
  const changeGestureMode = () => {
    if (gestureModeFixed) return;
    setHeadline("");
    setTopK([]);
    setStable(false);
    setSentence("");
    lastCommittedRef.current = "";

    if (gestureMode) {
      stopSendLoop();
      setPaused(false);
      stopWordsSession();
    } else {
      stopLettersLoop();
    }

    setGestureMode((g) => !g);
  };

  const clearSentence = () => {
    lastCommittedRef.current = "";
    const ws = wsRef.current;
    if (gestureMode && ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "clear_sentence" }));
    }
    setSentence("");
  };
  const undoWord = () => {
    const ws = wsRef.current;
    if (gestureMode && ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "undo" }));
    }
  };

  return (
    <div>
      {/* Video */}
      <div className={videoFrameCls}>
        <video ref={videoRef} autoPlay playsInline width="640" height="400" />
      </div>

      {/* Toolbar */}
      <div className={`flex items-center gap-3 rounded-lg px-4 py-3 ${softPanelCls}`}>
        <span
          onClick={() => {
            if (gestureMode && connected) {
              paused ? resumeWords() : pauseWords();
            }
          }}
          title={
            gestureMode
              ? (connected ? (paused ? "Click to resume" : "Click to pause") : "Not connected")
              : "Pause is only for Words mode"
          }
          className={`${pillBase} cursor-${gestureMode && connected ? "pointer" : "default"} ${connected ? (gestureMode ? (paused ? pillPaused : pillConnected) : pillConnected) : pillOffline}`}
        >
          {connected ? (gestureMode ? (paused ? "Paused" : `Connected (${status})`) : "Letters (Polling)") : "Offline"}
        </span>

        {!gestureModeFixed && (
          <button onClick={changeGestureMode} className={btnNeutral}>
            <img src={gestureMode ? gestureIcon : letterIcon} className="w-8 h-8" alt="Mode Toggle" />
          </button>
        )}

        <button onClick={undoWord} className={btnNeutral} disabled={!gestureMode}>
          Undo
        </button>
        <button onClick={clearSentence} className={btnNeutral} disabled={!gestureMode}>
          Clear
        </button>

        <div className="flex-1" />

        <button onClick={toggleSpeak} className={btnNeutral}>
          <img
            src={SoundOnIcon}
            className={`w-8 h-8 ${soundOn ? "" : "opacity-40"}`}
            alt="Speaker"
          />
        </button>
      </div>

      {/* Headline / TopK */}
      <div className={`mt-3 rounded-lg p-3 ${cardCls}`}>
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold">
            {gestureMode ? (headline ? `${headline}${stable ? " ✅" : ""}` : "—") : (headline || "—")}
          </h2>
          {gestureMode && (
            <div className={`text-sm ${textMutedCls}`}>
              {topK.map((t, i) => (
                <span key={i} className="mr-3">
                  {i + 1}. {t.label} {(t.p * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Sentence */}
      <div className="mt-3">
  <h3 className="text-lg font-semibold mb-1">Sentence</h3>
  
  <div className="flex rounded min-h-[60px]">
    <div className={`p-3 text-2xl w-4/5 ${boxCls}`}>
      {gestureMode ? (sentence || " ") : " "}
    </div>

    <div className="w-1/5 flex items-center justify-center">
      <button onClick={() => (toEnglish(sentence))} className="px-3 py-2 bg-blue-500 text-white rounded">
        To English
      </button>
    </div>
  </div>
</div>
    </div>
  );
};

export default TranslatorCamera;
