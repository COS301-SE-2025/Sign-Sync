// src/hooks/useSignStream.js
import { useCallback, useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker, HandLandmarker } from "@mediapipe/tasks-vision";

import pose_task from "../assets/pose_landmarker_full.task";
import hand_task from "../assets/hand_landmarker.task";

const WORDS_API_BASE   = "http://localhost:8007/api/stt";
const LETTERS_API_BASE = "http://localhost:8007/api/alphabet";
const GRAMMAR_API_BASE = "http://localhost:8007/api/word";

const SEND_INTERVAL_MS    = 80;
const LETTERS_INTERVAL_MS = 500;

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

export function useSignStream({ mode = "words", onPrediction, autoStart = true, videoConstraints = { width: 640, height: 400 } } = {}) {
  // Refs to engine pieces
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const poseRef = useRef(null);
  const handRef = useRef(null);
  const loopTimerRef = useRef(null);
  const sessionIdRef = useRef(null);

  // Public state
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [paused, setPaused] = useState(false);

  const [headline, setHeadline] = useState("");
  const [topK, setTopK] = useState([]);    // [{label,p}]
  const [stable, setStable] = useState(false);
  const [sentence, setSentence] = useState("");

  const [soundOn, setSoundOn] = useState(false);
  const lastCommittedRef = useRef("");

  // -------- helpers (speak / english) --------
  const speakText = useCallback((text) => {
    if (!("speechSynthesis" in window) || !text) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = "en-US";
    utt.onstart = () => setSoundOn(true);
    utt.onend = () => setSoundOn(false);
    utt.onerror = () => setSoundOn(false);
    window.speechSynthesis.speak(utt);
  }, []);

  const toggleSpeak = useCallback(() => {
    const text = (sentence || "").replace(/\s+/g, " ").trim();
    if (!text) {
      if (window.speechSynthesis.speaking) { window.speechSynthesis.cancel(); setSoundOn(false); }
      return;
    }
    if (window.speechSynthesis.speaking) { window.speechSynthesis.cancel(); setSoundOn(false); }
    else { speakText(text); }
  }, [sentence, speakText]);

  const toEnglish = useCallback(async (text) => {
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
    } catch {/* ignore */}
  }, [sentence]);

  // -------- words mode engine --------
  const tickSendWords = useCallback(() => {
    const videoEl = videoRef.current;
    const pose = poseRef.current;
    const hand = handRef.current;
    const ws = wsRef.current;
    if (!videoEl || !pose || !hand || !ws || ws.readyState !== WebSocket.OPEN) return;

    const ts = performance.now();

    // Pose
    const pRes = pose.detectForVideo(videoEl, ts);
    let pose33 = null;
    if (pRes?.landmarks?.length) {
      const first = pRes.landmarks[0];
      if (first && first.length === 33) pose33 = first.map((lm) => [lm.x, lm.y, lm.z ?? 0, lm.visibility ?? 0]);
    }
    if (!pose33) return;

    // Hands
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
    try { ws.send(JSON.stringify(payload)); } catch { /* ignore */ }
  }, []);

}
