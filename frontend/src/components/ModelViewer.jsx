import React from 'react'
import { Suspense, useRef, useState } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls, Grid, Environment } from '@react-three/drei'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import * as THREE from 'three'
import './ModelViewer.css'

// First-person camera controls
function FirstPersonControls() {
  const cameraRef = useRef(null)
  const [keys, setKeys] = useState({
    w: false,
    a: false,
    s: false,
    d: false,
  })

  useFrame((state) => {
    if (!cameraRef.current) return

    const camera = cameraRef.current
    const moveSpeed = 0.1
    const rotationSpeed = 0.02

    // Handle keyboard input
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

    // Add event listeners
    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('keyup', handleKeyUp)

    // Move camera based on keys
    const direction = new THREE.Vector3()
    camera.getWorldDirection(direction)

    if (keys.w) {
      camera.position.add(direction.multiplyScalar(moveSpeed))
    }
    if (keys.s) {
      camera.position.add(direction.multiplyScalar(-moveSpeed))
    }
    if (keys.a) {
      camera.position.add(new THREE.Vector3(-direction.z, 0, direction.x).multiplyScalar(moveSpeed))
    }
    if (keys.d) {
      camera.position.add(new THREE.Vector3(direction.z, 0, -direction.x).multiplyScalar(moveSpeed))
    }

    // Handle mouse look
    const handleMouseMove = (event) => {
      if (document.pointerLockElement === document.body) {
        const deltaX = event.movementX * rotationSpeed
        const deltaY = event.movementY * rotationSpeed

        camera.rotation.y -= deltaX
        camera.rotation.x -= deltaY
        camera.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, camera.rotation.x))
      }
    }

    document.addEventListener('mousemove', handleMouseMove)

    // Cleanup
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('keyup', handleKeyUp)
      document.removeEventListener('mousemove', handleMouseMove)
    }
  })

  return <primitive ref={cameraRef} object={state.camera} />
}

// Model loader component
function Model({ url }) {
  const model = useLoader(GLTFLoader, url)
  
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
  
  return (
    <group>
      <primitive object={model.scene} />
      {/* Add a simple test cube at origin */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color="red" />
      </mesh>
      {/* Add a wireframe box around the model for debugging */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[size.x, size.y, size.z]} />
        <meshBasicMaterial color="yellow" wireframe />
      </mesh>
    </group>
  )
}

// Fallback for OBJ files
function OBJModel({ url }) {
  const model = useLoader(OBJLoader, url)
  return <primitive object={model} />
}

// Demo model component (simple geometric shapes)
function DemoModel() {
  return (
    <group>
      {/* Room walls */}
      <mesh position={[0, 2, -5]}>
        <planeGeometry args={[10, 4]} />
        <meshStandardMaterial color="#f0f0f0" />
      </mesh>
      <mesh position={[-5, 2, 0]} rotation={[0, Math.PI / 2, 0]}>
        <planeGeometry args={[10, 4]} />
        <meshStandardMaterial color="#e0e0e0" />
      </mesh>
      <mesh position={[5, 2, 0]} rotation={[0, Math.PI / 2, 0]}>
        <planeGeometry args={[10, 4]} />
        <meshStandardMaterial color="#e0e0e0" />
      </mesh>
      
      {/* Floor */}
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[10, 10]} />
        <meshStandardMaterial color="#d0d0d0" />
      </mesh>
      
      {/* Furniture - Table */}
      <mesh position={[0, 0.5, -2]}>
        <boxGeometry args={[2, 1, 1]} />
        <meshStandardMaterial color="#8B4513" />
      </mesh>
      
      {/* Furniture - Chair */}
      <mesh position={[1.5, 0.3, -1]}>
        <boxGeometry args={[0.5, 0.6, 0.5]} />
        <meshStandardMaterial color="#654321" />
      </mesh>
      
      {/* Furniture - Lamp */}
      <mesh position={[-2, 1.5, -3]}>
        <cylinderGeometry args={[0.1, 0.1, 3]} />
        <meshStandardMaterial color="#FFD700" />
      </mesh>
      
      {/* Ceiling light */}
      <mesh position={[0, 3.8, 0]}>
        <sphereGeometry args={[0.3]} />
        <meshStandardMaterial color="#FFFF00" emissive="#FFFF00" emissiveIntensity={0.5} />
      </mesh>
    </group>
  )
}

export default function ModelViewer({ modelUrl, onReset }) {
  const [controlsMode, setControlsMode] = useState('orbit') // 'orbit' | 'firstperson'
  const [showGrid, setShowGrid] = useState(true)

  const isDemo = modelUrl === 'demo-cube'
  // Better file type detection - check for blob URLs and file extensions
  const isGLTF = modelUrl.toLowerCase().includes('.glb') || 
                 modelUrl.toLowerCase().includes('.gltf') ||
                 modelUrl.startsWith('blob:') // Blob URLs from file uploads

  return (
    <div className="model-viewer">
      <div className="viewer-controls">
        <div className="control-group">
          <button 
            className={`control-btn ${controlsMode === 'orbit' ? 'active' : ''}`}
            onClick={() => setControlsMode('orbit')}
          >
            🎯 Orbit
          </button>
          <button 
            className={`control-btn ${controlsMode === 'firstperson' ? 'active' : ''}`}
            onClick={() => setControlsMode('firstperson')}
          >
            🚶 First Person
          </button>
        </div>
        
        <div className="control-group">
          <button 
            className={`control-btn ${showGrid ? 'active' : ''}`}
            onClick={() => setShowGrid(!showGrid)}
          >
            📐 Grid
          </button>
          <button className="control-btn" onClick={onReset}>
            🔄 Reset
          </button>
        </div>
      </div>

      <div className="canvas-container">
        <Canvas camera={{ position: [5, 5, 5], fov: 75 }}>
          <Suspense fallback={null}>
            {isDemo ? (
              <DemoModel />
            ) : isGLTF ? (
              <Model url={modelUrl} />
            ) : (
              <OBJModel url={modelUrl} />
            )}
            
            {/* Test cube to verify 3D scene is working */}
            <mesh position={[0, 1, 0]}>
              <boxGeometry args={[0.5, 0.5, 0.5]} />
              <meshStandardMaterial color="blue" />
            </mesh>
          </Suspense>
          
          {showGrid && <Grid args={[20, 20]} />}
          
          <Environment preset="apartment" />
          
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          
          {controlsMode === 'orbit' ? (
            <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
          ) : (
            <FirstPersonControls />
          )}
        </Canvas>
      </div>

      {controlsMode === 'firstperson' && (
        <div className="controls-hint">
          <p>🎮 Click to lock mouse • WASD to move • Mouse to look around</p>
        </div>
      )}
    </div>
  )
}