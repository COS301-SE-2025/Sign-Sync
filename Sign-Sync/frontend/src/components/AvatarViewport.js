import React, {useEffect, useMemo, useRef, useState} from 'react';
import {Canvas, useFrame, useLoader, useThree} from '@react-three/fiber';
import { useGLTF, useAnimations, Text } from '@react-three/drei';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';
import PreferenceManager from "./PreferenceManager";

function Avatar({signs}) {
    const avatarReference = useRef();
    const {scene, animations, materials} = useGLTF(TranslatorAvatar);
    const {actions, mixer} = useAnimations(animations,avatarReference);
    const { camera } = useThree();
    const [translatedWord, setTranslatedWord] = useState("");
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
    const animationSpeed = 1.0  ; //User preference will set this value


    useEffect(() => {
        materials["Face-CM-Material"].map.offset.x = 0.0; // How to change emotions
        materials["Face-CM-Material"].map.offset.y = 0.25; // How to change emotions
        if (!actions["Idle"]) return;
        mixer.clipAction(actions["Idle"].getClip());
        actions["Idle"].reset().play();
    }, [scene,camera]);

    useEffect(() => {
            let animationIndex = [mixer.clipAction(actions["Idle"].getClip()), null];
            async function playAnimations() {
                for (let i = 0; i < signs.length; i++) {
                    const animation = actions[signs[i]];
                    animationIndex[1] = mixer.clipAction(animation.getClip());
                    animationIndex[1].reset();
                    if (animationIndex[0] !== null) {
                        animationIndex[1].fadeIn(0.2).play();
                        animationIndex[1].crossFadeFrom(animationIndex[0], 0.2, false);
                        animationIndex[1].timeScale = animationSpeed;
                    }
                    animationIndex[0] = animationIndex[1];
                    if (signs[i].includes("Pronoun-")) {
                        setTranslatedWord(signs[i].substring(signs[i].indexOf("-") + 1, signs[i].length));
                    } else {
                        setTranslatedWord(signs[i]);
                    }
                    await new Promise(resolve => setTimeout(resolve, (animation.getClip().duration / animationSpeed) * 1000));
                }
                if (animationIndex[0] !== null) {
                    animationIndex[0].fadeOut(0.5).stop();
                    mixer.clipAction(actions["Idle"].getClip());
                    actions["Idle"].reset().play();
                    setTranslatedWord("");
                }
            }
            playAnimations();
            return () => {
                mixer.stopAllAction();
                setTranslatedWord("");
            };
    },[signs]);

    return <>
        <primitive ref={avatarReference} object={scene} position={[-0.5,-2,3]}/>
        {<Text position={[1, -0.5, 3]} fontSize={0.3}  color = {isDarkMode ? "white": "black"}> {translatedWord} </Text>}
    </>;
}

export default function AvatarViewport({input,trigger}) {
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
                        alert(`Translation failed: ${errorData.message}`);
                        console.error("Translation error:", errorData);
                    }
                }catch(error)
                {
                    console.error("Error during Translation:", error);
                    alert("An error occurred during Translation. Please try again.");
                }
            }
            setSigns(signs);
        }

        if(input.length > 0){
            SignAPI();
        }
    }, [input,trigger]);

    return (
        <Canvas orthographic camera={{position: [0,0,4.5], zoom: 200}} style={{ height: '65vh',width:'130vh', background: isDarkMode ? '#36454f' : '#e5e7eb'}}>
            <Avatar signs={signs}/>
            <directionalLight color="white" position={[5,10,7.5]} intensity={1}/>
            <ambientLight color="white" intensity={0.75}/>
        </Canvas>
    );
}