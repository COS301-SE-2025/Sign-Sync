import React, {useEffect, useRef} from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei';
import {MeshStandardMaterial, Color, DirectionalLight, AmbientLight} from 'three';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';

function Avatar({input}) {
    console.log({input});
    const avatarReference = useRef();
    const {scene, animations} = useGLTF(TranslatorAvatar);
    const {actions} = useAnimations(animations,avatarReference);
    useEffect(() => {
        const sun = new DirectionalLight('rgb(255,255,255)',1);
        sun.position.set(5,10,7.5);
        scene.add(sun);
        scene.add(new AmbientLight(0xffffff,0.75));
        actions['NodsHead']?.play();
    }, [actions]);

    return <primitive ref={avatarReference} object={scene} position={[0,-2,3]} />;
}

export default function AvatarViewport({input}) {
    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <Avatar input={input}/>
        </Canvas>
    );
}
