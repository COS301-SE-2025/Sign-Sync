import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { MeshStandardMaterial, Color } from 'three';
function Avatar() {
    const avatarReference = useRef();
    return (
        <mesh ref={avatarReference} position={[0,0,0]}>
        </mesh>
    );
}

export default function AvatarViewport() {
    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[5,5,5]} />
            <Avatar />
        </Canvas>
    );
}
