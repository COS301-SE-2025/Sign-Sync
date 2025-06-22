import React, {useEffect, useMemo, useRef} from 'react';
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

    const signs = useMemo(() => {//Stops resubmitting the sentence
        const arr = [];
        const words = input.split(' ');

        //FastAPI Goes here but this is just for testing
        for (let i = 0; i < words.length; i++) {
            if(words[i] === "nod")arr.push("Nod");
            if(words[i] === "shake") arr.push("Shake");
        }
        return arr;

    }, [input]);

    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <Avatar signs={signs}/>
        </Canvas>
    );
}