import React from 'react'
import { useState } from 'react'
import VideoUpload from './components/VideoUpload'
import ModelViewer from './components/ModelViewer'
import './App.css'

function App() {
  const [status, setStatus] = useState('idle') // 'idle' | 'uploading' | 'processing' | 'complete' | 'error'
  const [modelUrl, setModelUrl] = useState(null)
  const [error, setError] = useState(null)

  const handleUploadComplete = (url) => {
    setModelUrl(url)
    setStatus('complete')
  }

  const handleError = (errorMessage) => {
    setError(errorMessage)
    setStatus('error')
  }

  const resetApp = () => {
    setStatus('idle')
    setModelUrl(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>nuu</h1>
        <h2>Scan it, plan it, buy it.</h2>
      </header>

      <main className="app-main">
        {status === 'idle' && (
          <div className="body">
            <VideoUpload
              onUploadStart={() => setStatus('uploading')}
              onProcessingStart={() => setStatus('processing')}
              onComplete={handleUploadComplete}
              onError={handleError}
            />
            <div className="app-instructions">
              <div className="instruction">
                <h3 className="instruction-title">1. UPLOAD</h3>
                <div>
                  <p>Drag and drop or upload a video of your room.</p>
                </div>
              </div>
              <div className="instruction">
                <h3 className="instruction-title">2. PROCESS</h3>
                <div>
                  <p>nuu will automatically process your video and generate a 3D model of your room.</p>
                </div>
              </div>
              <div className="instruction">
                <h3 className="instruction-title">3. VIEW</h3>
                <div>
                  <p>See your room in 3D from any angle and any perspective, whether you're on a walk outside or picking out furniture at a store.</p>
                </div>
              </div>
            </div>
          </div>

        )}

        {status === 'uploading' && (
          <div className="status-container">
            <div className="loading-spinner"></div>
            <p>Uploading video...</p>
          </div>
        )}

        {status === 'processing' && (
          <div className="status-container">
            <div className="loading-spinner"></div>
            <p>Processing frames...</p>
            <p className="status-sub">Generating 3D model...</p>
          </div>
        )}

        {status === 'complete' && modelUrl && (
          <ModelViewer modelUrl={modelUrl} onReset={resetApp} />
        )}

        {status === 'error' && (
          <div className="error-container">
            <h3>❌ Processing Failed</h3>
            <p>{error}</p>
            <button onClick={resetApp} className="reset-button">
              Try Again
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App