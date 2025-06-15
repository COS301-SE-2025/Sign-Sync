import React, {useEffect, useRef} from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, useAnimations } from '@react-three/drei';
import {DirectionalLight, AmbientLight} from 'three';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';

function Avatar({signs}) {
    const avatarReference = useRef();
    const {scene, animations} = useGLTF(TranslatorAvatar);
    const {actions} = useAnimations(animations,avatarReference);

    useEffect(() => {
        const sun = new DirectionalLight('rgb(255,255,255)',1);
        sun.position.set(5,10,7.5);
        scene.add(sun);
        scene.add(new AmbientLight(0xffffff,0.75));
    }, [scene]);

    useEffect(() => {
        if(signs.length > 0){
            for(let i = 0; i < signs.length; i++){
                actions[signs[i]]?.play();
                const frames = actions[signs[i]].getClip().duration;
                if(signs.length > 1) {
                    setTimeout(() => {
                        actions[signs[i]].stop();
                    }, frames * 1000);
                }
            }
        }
    }, [actions,signs]);

    return <primitive ref={avatarReference} object={scene} position={[0,-2,3]} />;
}

export default function AvatarViewport({input}) {
    const words = input.split(' ');
    const signs = [];
    //Fast API connection will happen here but this will be implemented later
    for (let i = 0; i < words.length; i++) {
        if(words[i] === "nod"){
            signs.push("NodsHead");

        }
        if(words[i] === "shake"){
            signs.push("ShakesHead");
        }
    }

    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <Avatar signs={signs}/>
        </Canvas>
    );
}