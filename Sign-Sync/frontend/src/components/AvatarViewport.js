import React, { useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei';
import {MeshStandardMaterial, Color, DirectionalLight} from 'three';
import TranslatorAvatar from '../assets/3DModels/Avatar.glb';

function Avatar() {
    const avatarReference = useRef();
    const {scene} = useGLTF(TranslatorAvatar);
    scene.add(new DirectionalLight())
    return <primitive ref={avatarReference} object={scene} position={[0,-2,3]} />;
}

export default function AvatarViewport() {
    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <ambientLight intensity={0.75} />
            <directionalLight position={[5,10,7.5]} />
            <Avatar />
        </Canvas>
    );
}
