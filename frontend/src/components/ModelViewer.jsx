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
  return <primitive object={model.scene} />
}

// Fallback for OBJ files
function OBJModel({ url }) {
  const model = useLoader(OBJLoader, url)
  return <primitive object={model} />
}

export default function ModelViewer({ modelUrl, onReset }) {
  const [controlsMode, setControlsMode] = useState('orbit') // 'orbit' | 'firstperson'
  const [showGrid, setShowGrid] = useState(true)

  const isGLTF = modelUrl.toLowerCase().includes('.glb') || modelUrl.toLowerCase().includes('.gltf')

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
            {isGLTF ? (
              <Model url={modelUrl} />
            ) : (
              <OBJModel url={modelUrl} />
            )}
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
