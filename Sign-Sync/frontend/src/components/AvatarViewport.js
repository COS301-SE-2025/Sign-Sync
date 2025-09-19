import React, {useEffect, useMemo, useRef, useState} from 'react';
import {Canvas, useFrame, useLoader, useThree} from '@react-three/fiber';
import { useGLTF, useAnimations, Text } from '@react-three/drei';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';
import PreferenceManager from "./PreferenceManager";
import * as THREE from "three";
import { toast } from "react-toastify";
import preferenceManager from "./PreferenceManager";


function Avatar({signs,emotion = "Neutral"}) {
    const avatarReference = useRef();
    const {scene, animations, materials} = useGLTF(TranslatorAvatar);
    const {actions, mixer} = useAnimations(animations,avatarReference);
    const { camera } = useThree();
    const [translatedWord, setTranslatedWord] = useState("");
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
    const animationSpeed = PreferenceManager.getPreferences().animationSpeed;
    const emotions = {"Neutral":[0,0],"Happy":[0,0.25],"Sad":[0,0.5],"Angry":[0,0.75],"Surprise":[0.5,0]};
    const emotionsRef = useRef(null);
    const animationController = useRef(null);
    const speeds = {"Very Slow": 0.75,"Slow":1,"Normal":1.5,"Fast":2.5,"Very Fast":5};
    useEffect(() => {
        setAvatarType(PreferenceManager.getPreferences().preferredAvatar);
        if (!actions["Idle"]) return;
        mixer.clipAction(actions["Idle"].getClip());
        actions["Idle"].reset().play();
    }, [scene,camera]);

    useEffect(() =>{
        if (emotion === "") emotion = "Neutral";
        emotionsRef.current = emotion;
        materials["Face-CM-Material"].map.offset.x = emotions[emotionsRef.current][0];
        materials["Face-CM-Material"].map.offset.y = emotions[emotionsRef.current][1];
        materials["Face-CM-Material"].map.needsUpdate = true;
    },[emotion])

    useEffect(() => {
        if (!actions["Idle"]) return;
        const controller = new AbortController();
        animationController.current = controller;
        let animationIndex = [mixer.clipAction(actions["Idle"].getClip()), null];
        async function playAnimations() {
            try {
                for (let i = 0; i < signs.length; i++) {
                    const animation = actions[signs[i]];
                    animationIndex[1] = mixer.clipAction(animation.getClip());
                    animationIndex[1].reset();

                    if (animationIndex[0] !== null) {
                        animationIndex[1].fadeIn(0.2).play();
                        animationIndex[1].crossFadeFrom(animationIndex[0], 0.2, false);
                        animationIndex[1].timeScale = speeds[animationSpeed];
                    }
                    animationIndex[0] = animationIndex[1];
                    if (signs[i].includes("Pronoun-")) {
                        setTranslatedWord(signs[i].substring(signs[i].indexOf("-") + 1, signs[i].length));
                    } else {
                        setTranslatedWord(signs[i]);
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
                }
            }catch (error) {
                if (error.message !== 'Animation stopped - Rerun') {
                    console.error("Animation error other than rerun:", error);
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
        };
    },[signs,actions,mixer]);

    function setVisibility(visible) {
        scene.getObjectByName("Head-F").visible = !visible;
        scene.getObjectByName("Body-F").visible = !visible;
        scene.getObjectByName("Body-M").visible = visible;
        scene.getObjectByName("Head-M").visible = visible;
        scene.getObjectByName("Hair-CM").visible = false;
        scene.getObjectByName("Hair-CF").visible = false;
    }

    function setAvatarType(type) {
        console.log(type)
        switch (type) {
            case "Zac":
                setVisibility(true);
                scene.getObjectByName("Hair-CM").visible = true;
                break;
            case "Jenny":
                setVisibility(false);
                scene.getObjectByName("Hair-CF").visible = true;
                break;
        }
    }

    return <>
        <primitive ref={avatarReference} object={scene} position={[-0.5,-2,3]}/>
        {<Text position={[1, -0.5, 3]} fontSize={0.3}  color = {isDarkMode ? "white": "black"}> {translatedWord} </Text>}
    </>;
}

export default function AvatarViewport({input,emotion = "Neutral",trigger, height, width}) {
    const [signs,setSigns] = useState([]);
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

    useEffect(() => {
        async function SignAPI() {
            const words = input.toLowerCase().split(' ');
            console.log("Words:", words);
            let signs = [];
            for (let i = 0; i < words.length; i++) {
                try {
                    const response = await fetch('/textSignApi/getAnimation', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({word: words[i]}),
                    });
                    if (response.ok) {
                        const sign = await response.json();
                        if (Array.isArray(sign.response)) {
                            signs.push(...sign.response);
                        }else{
                            signs.push(sign.response);
                        }
                    } else {
                        const errorData = await response.json();
                        toast.error(`Translation failed: ${errorData.message}`);
                        console.error("Translation error:", errorData);
                    }
                }catch(error)
                {
                    console.error("Error during Translation:", error);
                    toast.error("An error occurred during Translation. Please try again.");
                }
            }
            setSigns(signs);
        }

        if(input.length > 0){
            SignAPI();
        }
    }, [input,trigger]);

    return (
        <div style={{ height: `${height}px`, width: `${width}px`, maxWidth: '100%', margin: '0 auto', background: isDarkMode ? '#36454f' : '#e5e7eb', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {/* <Canvas orthographic camera={{position: [0,0,4.5], zoom: 200}} style={{ height: height, width: width, background: isDarkMode ? '#36454f' : '#e5e7eb'}}> */}
            <Canvas orthographic camera={{position: [0,0,4.5], zoom: 200}} style={{ height: '100%', width: '100%', display: 'block'}}>
                <Avatar signs={signs} emotion={emotion} />
                <directionalLight color="white" position={[5,10,7.5]} intensity={1}/>
                <ambientLight color="white" intensity={0.75}/>
            </Canvas>
        </div>
    );
}