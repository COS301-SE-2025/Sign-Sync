import React, {useEffect, useMemo, useRef, useState} from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, useAnimations } from '@react-three/drei';
import {DirectionalLight, AmbientLight, AnimationMixer} from 'three';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';

function Avatar({signs}) {
    const avatarReference = useRef();
    const {scene, animations} = useGLTF(TranslatorAvatar);
    const {actions, mixer} = useAnimations(animations,avatarReference);

    useEffect(() => {
        const sun = new DirectionalLight('rgb(255,255,255)',1);
        sun.position.set(5,10,7.5);
        scene.add(sun);
        scene.add(new AmbientLight(0xffffff,0.75));
    }, [scene]);

    useEffect(() => {
        let animationIndex = [null,null];
        async function playAnimations (){
            for (let i = 0; i < signs.length; i++) {
                const animation = actions[signs[i]];
                animationIndex[1] = mixer.clipAction(animation.getClip());
                if(animationIndex[0]!==null){
                    animationIndex[0].stop();
                }
                animationIndex[0] = animationIndex[1];
                animationIndex[1].reset().play();
                await new Promise(resolve => setTimeout(resolve, animation.getClip().duration * 1000));
            }
            if(animationIndex[0]!==null){
                animationIndex[0].stop();
            }
        }
        playAnimations();
        return () => {mixer.stopAllAction();};
    },[signs]);

    return <primitive ref={avatarReference} object={scene} position={[0,-2,3]} />;
}

export default function AvatarViewport({input}) {
    const [signs,setSigns] = useState([]);
    useEffect(() => {
        async function SignAPI() {
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
    }, [input]);

    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <Avatar signs={signs}/>
        </Canvas>
    );
}