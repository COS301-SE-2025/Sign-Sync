import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { useGLTF, useAnimations, Text } from '@react-three/drei';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';
import PreferenceManager from "./PreferenceManager";
import * as THREE from "three";
import { toast } from "react-toastify";
import preferenceManager from "./PreferenceManager";


function Avatar({ onPlayingChange, signs, emotion = "Neutral" }) {
    const avatarReference = useRef();
    const { scene, animations, materials } = useGLTF(TranslatorAvatar);
    const { actions, mixer } = useAnimations(animations, avatarReference);
    const { camera } = useThree();
    const [translatedWord, setTranslatedWord] = useState("");
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
    const animationSpeed = PreferenceManager.getPreferences().animationSpeed;
    const avatarName = PreferenceManager.getPreferences().preferredAvatar
    const emotions = { "Neutral": [0, 0], "Happy": [0, 0.25], "Sad": [0, 0.5], "Anger": [0, 0.75], "Surprise": [0.5, 0] };
    const emotionsRef = useRef(null);
    const animationController = useRef(null);
    const speeds = { "Very Slow": 0.75, "Slow": 1, "Normal": 1.5, "Fast": 2.5, "Very Fast": 5 };

    const seq = Array.isArray(signs) ? signs : [];   //***********

    useEffect(() => {
        setAvatarType(avatarName);
        if (!actions["Idle"]) return;
        mixer.clipAction(actions["Idle"].getClip());
        actions["Idle"].reset().play();
    }, [scene, camera]);

    useEffect(() => {
        if (emotion === "") emotion = "Neutral";

        //Remove this when error is found
        if (emotion === "SSurprise") emotion = "Surprise";

        emotionsRef.current = emotion;
        materials[avatarName + "-Face"].map.offset.x = emotions[emotionsRef.current][0];
        materials[avatarName + "-Face"].map.offset.y = emotions[emotionsRef.current][1];
        materials[avatarName + "-Face"].map.needsUpdate = true;
    }, [emotion])

    useEffect(() => {
        if (!actions["Idle"]) return;
        const controller = new AbortController();
        animationController.current = controller;
        let animationIndex = [mixer.clipAction(actions["Idle"].getClip()), null];

        async function playAnimations() {
            try {
                if (seq.length === 0) {
                    actions["Idle"]?.reset()?.play();
                    setTranslatedWord("");
                    onPlayingChange?.(false);
                    return;
                }
                onPlayingChange?.(true);
                // for (let i = 0; i < signs.length; i++) {
                //     const animation = actions[signs[i]];
                for (let i = 0; i < seq.length; i++) {
                    const animation = actions[seq[i]];
                    if (!animation) continue;
                    animationIndex[1] = mixer.clipAction(animation.getClip());
                    animationIndex[1].reset();

                    if (animationIndex[0] !== null) {
                        animationIndex[1].fadeIn(0.2).play();
                        animationIndex[1].crossFadeFrom(animationIndex[0], 0.2, false);
                        animationIndex[1].timeScale = speeds[animationSpeed];
                    }

                    animationIndex[0] = animationIndex[1];

                    // if (signs[i].includes("Pronoun-")) {
                    //     setTranslatedWord(signs[i].substring(signs[i].indexOf("-") + 1, signs[i].length));
                    // } else {
                    //     setTranslatedWord(signs[i]);
                    // }
                    if (seq[i].includes("Pronoun-")) {
                        setTranslatedWord(seq[i].substring(seq[i].indexOf("-") + 1));
                    } else {
                        setTranslatedWord(seq[i]);
                    }

                    await new Promise((resolve, reject) => {
                        const animationTime = (animation.getClip().duration / speeds[animationSpeed]) * 1000;
                        const timer = setTimeout(resolve, animationTime);
                        controller.signal.addEventListener('abort', () => {
                            clearTimeout(timer);
                            reject(new Error('Animation stopped - Rerun'));
                        });
                    });
                }

                if (animationIndex[0] !== null) {
                    animationIndex[0].fadeOut(0.5).stop();
                    mixer.clipAction(actions["Idle"].getClip());
                    actions["Idle"].reset().play();
                    setTranslatedWord("");
                    emotionsRef.current = "Neutral";
                    materials[avatarName + "-Face"].map.offset.x = emotions[emotionsRef.current][0];
                    materials[avatarName + "-Face"].map.offset.y = emotions[emotionsRef.current][1];
                    materials[avatarName + "-Face"].map.needsUpdate = true;
                    onPlayingChange?.(false);
                }
            } catch (error) {
                if (error.message !== 'Animation stopped - Rerun') {
                    console.error("Animation error other than rerun:", error);
                    onPlayingChange?.(false);
                }
            } finally {
                if (animationController.current === controller) {
                    animationController.current = null;
                }
            }
        }

        playAnimations();

        return () => {
            controller.abort();
            mixer.stopAllAction();
            setTranslatedWord("");
            if (actions["Idle"]) {
                actions["Idle"].reset().play();
            }
            if (animationController.current === controller) {
                animationController.current = null;
            }
            onPlayingChange?.(false);
        };
        // },[signs,actions,mixer]);
    }, [seq, actions, mixer]);

    function setAvatarType(type) {
        scene.getObjectByName("Temp-Face").visible = false;
        scene.getObjectByName("Temp-Head").visible = false;
        scene.getObjectByName("Temp-Body").visible = false;
        switch (type) {
            case "Zac":
                scene.getObjectByName("Body-F").visible = false;
                scene.getObjectByName("Body-M").visible = true;
                scene.getObjectByName("Head-F").visible = false;
                scene.getObjectByName("Head-M").visible = true;
                scene.getObjectByName("Face-F").visible = false;
                scene.getObjectByName("Face-M").visible = true;
                scene.getObjectByName("Hair-CM").visible = true;
                scene.getObjectByName("Hair-CF").visible = false;
                scene.getObjectByName("Hair-Bongani").visible = false;
                scene.getObjectByName("Body-M").material = materials["Zac_BODY-Material"];
                scene.getObjectByName("Face-M").material = materials["Zac-Face"];
                scene.getObjectByName("Head-M").material = materials["Head"];
                break;
            case "Jenny":
                scene.getObjectByName("Body-F").visible = true;
                scene.getObjectByName("Body-M").visible = false;
                scene.getObjectByName("Head-F").visible = true;
                scene.getObjectByName("Head-M").visible = false;
                scene.getObjectByName("Face-F").visible = true;
                scene.getObjectByName("Face-M").visible = false;
                scene.getObjectByName("Hair-CM").visible = false;
                scene.getObjectByName("Hair-CF").visible = true;
                scene.getObjectByName("Hair-Bongani").visible = false;
                scene.getObjectByName("Body-F").material = materials["Zac_BODY-Material"];
                scene.getObjectByName("Face-F").material = materials["Jenny-Face"];
                scene.getObjectByName("Head-F").material = materials["Head"];
                break;
            case "Bongani":
                scene.getObjectByName("Body-F").visible = false;
                scene.getObjectByName("Body-M").visible = true;
                scene.getObjectByName("Head-F").visible = false;
                scene.getObjectByName("Head-M").visible = true;
                scene.getObjectByName("Face-F").visible = false;
                scene.getObjectByName("Face-M").visible = true;
                scene.getObjectByName("Hair-CM").visible = false;
                scene.getObjectByName("Hair-CF").visible = false;
                scene.getObjectByName("Hair-Bongani").visible = true;
                scene.getObjectByName("Body-M").material = materials["Bongani_BODY-Material"];
                scene.getObjectByName("Face-M").material = materials["Bongani-Face"];
                scene.getObjectByName("Head-M").material = materials["Bongani-Head"];
                break;
        }
    }

    return <>
        <primitive ref={avatarReference} object={scene} position={[0, -2, 3]} />
        {/* {<Text position={[1, -0.5, 3]} fontSize={0.3}  color = {isDarkMode ? "white": "black"}> {translatedWord} </Text>} */}
    </>;
}

export default function AvatarViewport({playing, input, emotion = "Neutral", trigger, height, width }) {
    const [signs, setSigns] = useState([]);
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

    useEffect(() => {

        const safeInput = typeof input === "string" ? input.trim() : "";
        if (!safeInput) {
            setSigns([]);
            return;
        }

        async function SignAPI() {
            // const words = input.toLowerCase().split(' ');
            const words = safeInput.toLowerCase().split(/\s+/);
            console.log("Words:", words);
            let signs = [];

            for (let i = 0; i < words.length; i++) {
                try {
                    const response = await fetch('/textSignApi/getAnimation', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ word: words[i] }),
                    });

                    if (response.ok) {
                        const sign = await response.json();
                        if (Array.isArray(sign.response)) {
                            signs.push(...sign.response);
                        } else if (sign.response) {
                            signs.push(sign.response);
                        }
                    } else {
                        // const errorData = await response.json();
                        // toast.error(`Translation failed: ${errorData.message}`);
                        // console.error("Translation error:", errorData);

                        // Safe read of error JSON (may be empty on 429)             
                        let msg = "";
                        try { msg = (await response.json())?.message || ""; } catch { }
                        if (response.status === 429) {
                            const ra = response.headers.get("Retry-After");
                            const secs = ra && /^\d+$/.test(ra) ? parseInt(ra, 10) : 30;
                            toast.error(msg || `Too many requests. Try again in ${secs}s.`);
                            signs = []; // nothing to play this round              
                            break;
                        } else {
                            toast.error(msg || "Translation failed.");
                        }
                    }
                } catch (error) {
                    console.error("Error during Translation:", error);
                    toast.error("An error occurred during Translation. Please try again.");
                }
            }
            setSigns(signs);
        }

        // if (input.length > 0) {
        if (safeInput.length > 0) {
            SignAPI();
        }
    }, [input, trigger]);

    return (
        <div style={{ height: `${height}px`, width: `${width}px`, maxWidth: '100%', margin: '0 auto', background: isDarkMode ? '#1B2432' : '#e5e7eb', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {/* <Canvas orthographic camera={{position: [0,0,4.5], zoom: 200}} style={{ height: height, width: width, background: isDarkMode ? '#36454f' : '#e5e7eb'}}> */}
            <Canvas orthographic camera={{ position: [0, 0, 4.5], zoom: 200 }} style={{ height: '100%', width: '100%', display: 'block' }}>
                <Avatar signs={signs} emotion={emotion} onPlayingChange={playing}/>
                <directionalLight color="white" position={[5, 10, 7.5]} intensity={1} />
                <ambientLight color="white" intensity={0.75} />
            </Canvas>
        </div>
    );
}