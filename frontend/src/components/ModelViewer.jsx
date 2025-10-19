import React from 'react'
import { Suspense, useRef, useState, useEffect } from 'react'
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber'
import { OrbitControls, Grid, Environment } from '@react-three/drei'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import * as THREE from 'three'
import './ModelViewer.css'

// WASD keyboard controls to complement OrbitControls
function WASDControls() {
    const { camera } = useThree()
    const [keys, setKeys] = useState({
        w: false,
        a: false,
        s: false,
        d: false,
    })

    useEffect(() => {
        const handleKeyDown = (event) => {
            switch (event.key.toLowerCase()) {
                case 'w':
                    setKeys(prev => ({ ...prev, w: true }))
                    break
                case 'a':
                    setKeys(prev => ({ ...prev, a: true }))
                    break
                case 's':
                    setKeys(prev => ({ ...prev, s: true }))
                    break
                case 'd':
                    setKeys(prev => ({ ...prev, d: true }))
                    break
            }
        }

        const handleKeyUp = (event) => {
            switch (event.key.toLowerCase()) {
                case 'w':
                    setKeys(prev => ({ ...prev, w: false }))
                    break
                case 'a':
                    setKeys(prev => ({ ...prev, a: false }))
                    break
                case 's':
                    setKeys(prev => ({ ...prev, s: false }))
                    break
                case 'd':
                    setKeys(prev => ({ ...prev, d: false }))
                    break
            }
        }

        document.addEventListener('keydown', handleKeyDown)
        document.addEventListener('keyup', handleKeyUp)

        return () => {
            document.removeEventListener('keydown', handleKeyDown)
            document.removeEventListener('keyup', handleKeyUp)
        }
    }, [])

    useFrame(() => {
        const moveSpeed = 0.2
        
        // Get camera's forward direction (including vertical component)
        const forward = new THREE.Vector3()
        camera.getWorldDirection(forward)
        forward.normalize()

        // Get camera's right direction
        const right = new THREE.Vector3()
        right.crossVectors(forward, camera.up).normalize()

        if (keys.w) {
            camera.position.add(forward.clone().multiplyScalar(moveSpeed))
        }
        if (keys.s) {
            camera.position.add(forward.clone().multiplyScalar(-moveSpeed))
        }
        if (keys.a) {
            camera.position.add(right.clone().multiplyScalar(-moveSpeed))
        }
        if (keys.d) {
            camera.position.add(right.clone().multiplyScalar(moveSpeed))
        }
    })

    return null
}

// Model loader component
function Model({ url }) {
    const model = useLoader(GLTFLoader, url)
    const [isSetup, setIsSetup] = useState(false)

    // Reset setup state when URL changes
    useEffect(() => {
        setIsSetup(false)
    }, [url])

    useEffect(() => {
        if (model && model.scene && !isSetup) {
            console.log('Setting up model after load...')

            // Debug logging
            console.log('GLB Model loaded:', model)
            console.log('Model scene:', model.scene)
            console.log('Model scene children:', model.scene.children)

            // Calculate bounding box to center and scale the model
            const box = new THREE.Box3().setFromObject(model.scene)
            const center = box.getCenter(new THREE.Vector3())
            const size = box.getSize(new THREE.Vector3())

            console.log('Model bounding box:', { center, size })
            console.log('Model position before:', model.scene.position)
            console.log('Model scale before:', model.scene.scale)

            // Reset position and scale first
            model.scene.position.set(0, 0, 0)
            model.scene.scale.set(1, 1, 1)

            // Center the model
            model.scene.position.sub(center)

            // Scale the model to be much smaller and more manageable
            const maxDimension = Math.max(size.x, size.y, size.z)
            if (maxDimension > 0) {
                const scale = 2 / maxDimension  // Much smaller scale - target 2 units max
                model.scene.scale.setScalar(scale)
                console.log('Model scaled by:', scale)
                console.log('Original size:', size)
                console.log('Scaled size:', {
                    x: size.x * scale,
                    y: size.y * scale,
                    z: size.z * scale
                })
            }

            // Position the model at a better location for viewing
            model.scene.position.set(0, 0, 0)  // Reset to origin

            console.log('Model position after:', model.scene.position)
            console.log('Model scale after:', model.scene.scale)

            // Check if model has materials and make them more visible
            model.scene.traverse((child) => {
                if (child.isMesh) {
                    console.log('Found mesh:', child.name, 'Material:', child.material)
                    if (child.material) {
                        console.log('Material type:', child.material.type)
                        console.log('Material color:', child.material.color)

                        // Make the material more visible
                        child.material.color.setHex(0x00ff00) // Bright green
                        child.material.needsUpdate = true
                        console.log('Changed material color to green')
                    }
                }
            })

            setIsSetup(true)
        }
    }, [model, isSetup])

    return <primitive object={model.scene} />
}

// Fallback for OBJ files
function OBJModel({ url }) {
    const model = useLoader(OBJLoader, url)
    return <primitive object={model} />
}

export default function ModelViewer({ modelUrl, onReset }) {
    const [showGrid, setShowGrid] = useState(true)

    const isDemo = modelUrl === 'demo-cube'
    // Better file type detection - check for blob URLs and file extensions
    const isGLTF = modelUrl.toLowerCase().includes('.glb') ||
        modelUrl.toLowerCase().includes('.gltf') ||
        modelUrl.startsWith('blob:') // Blob URLs from file uploads

    return (
        <div className="model-viewer">
            <div className="viewer-controls">
                <button className="control-btn home-btn" onClick={onReset}>
                    Return to Home
                </button>

                <div className="control-group">
                    <button
                        className={`control-btn ${showGrid ? 'active' : ''}`}
                        onClick={() => setShowGrid(!showGrid)}
                    >
                        Toggle Grid
                    </button>
                </div>

            </div>

            <div className="canvas-container">
                <Canvas camera={{ position: [5, 5, 5], fov: 75 }}>
                    <Suspense fallback={null}>
                        {isDemo ? (
                            <DemoModel />
                        ) : isGLTF ? (
                            <Model key={modelUrl} url={modelUrl} />
                            ) : (
                                <OBJModel key={modelUrl} url={modelUrl} />
                            )}
                        </Suspense>

                    {showGrid && <Grid args={[20, 20]} />}

                    <Environment preset="apartment" />

                    <ambientLight intensity={0.5} />
                    <directionalLight position={[10, 10, 5]} intensity={1} />

                    <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
                    <WASDControls />
                </Canvas>
            </div>
        </div>
    )
}