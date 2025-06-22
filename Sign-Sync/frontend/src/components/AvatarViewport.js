import React, {useEffect, useMemo, useRef, useState} from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, useAnimations } from '@react-three/drei';
import {DirectionalLight, AmbientLight, AnimationMixer} from 'three';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';

function Avatar({signs}) {
    const avatarReference = useRef();
    const {scene, animations} = useGLTF(TranslatorAvatar);
    const {actions} = useAnimations(animations,avatarReference);
    const animationSequencer = new AnimationMixer(avatarReference.current);

    useEffect(() => {
        const sun = new DirectionalLight('rgb(255,255,255)',1);
        sun.position.set(5,10,7.5);
        scene.add(sun);
        scene.add(new AmbientLight(0xffffff,0.75));
    }, [scene]);

    useEffect(() => {
        console.log(signs);
    },[signs]);

    return <primitive ref={avatarReference} object={scene} position={[0,-2,3]} />;
}

export default function AvatarViewport({input}) {

    const [signs,setSigns] = useState([]);
    useEffect(() => {
        async function API() {
            const words = input.split(' ');
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
                            for (let i = 0; i < sign.response.length; i++) {
                                signs.push(sign.response[i]);
                            }
                        }else{
                            signs.push(sign.response);
                        }
                        console.log(sign.response);
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
        if(input.length >= 0){
            API();
        }
    }, [input]);

    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <Avatar signs={signs}/>
        </Canvas>
    );
}