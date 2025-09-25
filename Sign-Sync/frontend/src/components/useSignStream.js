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

}
