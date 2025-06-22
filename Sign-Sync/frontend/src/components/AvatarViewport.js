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

        handleSubmit = async (e) =>
        {
            e.preventDefault();

            if(!this.validateForm()) return;

            const { email, password } = this.state;

            try
            {
                const response = await fetch('/userApi/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ word: words }),
                });

                if(response.ok) {
                    await response.json();
                }
                else
                {
                    const errorData = await response.json();
                    alert(`Translation failed: ${errorData.message}`);
                    console.error("Translation error:", errorData);
                }
            }
            catch(error) {
                console.error("Error during Translation:", error);
                alert("An error occurred during Translation. Please try again.");
            }
        };
        return arr;

    }, [input]);

    return (
        <Canvas style={{ height: '50vh', background: '#222' }}>
            <Avatar signs={signs}/>
        </Canvas>
    );
}