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
        <h1>🐌 Nuu</h1>
        <p>Scan your room and carry your home with you</p>
      </header>
      
      <main className="app-main">
        {status === 'idle' && (
          <VideoUpload 
            onUploadStart={() => setStatus('uploading')}
            onProcessingStart={() => setStatus('processing')}
            onComplete={handleUploadComplete}
            onError={handleError}
          />
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