import React, { useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei';
import {MeshStandardMaterial, Color, DirectionalLight, AmbientLight} from 'three';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';

function Avatar() {
    const avatarReference = useRef();
    const {scene} = useGLTF(TranslatorAvatar);
    const sun = new DirectionalLight('rgb(255,255,255)',1);
    sun.position.set(5,10,7.5);
    scene.add(sun);
    scene.add(new AmbientLight(0xffffff,0.75));
    return <primitive ref={avatarReference} object={scene} position={[0,-2,3]} />;
}

export default function AvatarViewport() {
    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <Avatar />
        </Canvas>
    );
}
