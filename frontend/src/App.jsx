import React from 'react'
import { useState } from 'react'
import VideoUpload from './components/VideoUpload'
import ModelViewer from './components/ModelViewer'
import { getApiUrl } from './config/api'
import './App.css'

function App() {
  const [status, setStatus] = useState('idle') // 'idle' | 'uploading' | 'processing' | 'complete' | 'error'
  const [modelUrl, setModelUrl] = useState(null)
  const [error, setError] = useState(null)
  const [jobId, setJobId] = useState(null) // Added jobId state
  const [processingMessage, setProcessingMessage] = useState('') // Added processingMessage state

  // Backend to frontend status mapping:
  // 'uploaded' -> 'complete' (upload successful)
  // 'extracting_frames' | 'estimating_depth' | 'reconstructing_3d' -> 'processing'
  // 'completed' -> 'complete' (processing done)
  // 'failed' -> 'error'

  // Poll backend for job status
  const pollJobStatus = async (jobId) => {
    try {
      const response = await fetch(getApiUrl(`/api/status/${jobId}`))
      const data = await response.json()
      
      console.log('Job status:', data.status)
      
      switch (data.status) {
        case 'uploaded':
          setStatus('complete')
          setProcessingMessage('Video uploaded successfully!')
          break
          
        case 'extracting_frames':
          setStatus('processing')
          setProcessingMessage('Extracting frames from video...')
          break
          
        case 'estimating_depth':
          setStatus('processing')
          setProcessingMessage('Estimating depth information...')
          break
          
        case 'reconstructing_3d':
          setStatus('processing')
          setProcessingMessage('Reconstructing 3D model...')
          break
          
        case 'completed':
          setStatus('complete')
          setModelUrl(data.modelUrl)
          setProcessingMessage('3D model ready!')
          break
          
        case 'failed':
          setStatus('error')
          setError(data.error || 'Processing failed')
          break
          
        default:
          // Still processing, poll again in 2 seconds
          setTimeout(() => pollJobStatus(jobId), 2000)
      }
    } catch (error) {
      console.error('Error polling job status:', error)
      setStatus('error')
      setError('Failed to check processing status')
    }
  }

  const handleUploadComplete = (url) => {
    // For direct 3D model uploads (GLB/OBJ files)
    setModelUrl(url)
    setStatus('complete')
  }

  const handleJobCreated = (jobId) => {
    setJobId(jobId)
    setStatus('processing')
    setProcessingMessage('Starting processing...')
    // Start polling for status updates
    pollJobStatus(jobId)
  }

  const handleError = (errorMessage) => {
    setError(errorMessage)
    setStatus('error')
  }

  const resetApp = () => {
    setStatus('idle')
    setModelUrl(null)
    setError(null)
    setJobId(null) // Reset jobId
    setProcessingMessage('') // Reset processingMessage
  }

  return (
    <div className="app">
      <header className="header">
        <div className="app-header">
        <h1>nuu</h1>
        <h2>scan it, plan it, buy it.</h2>
        </div>
      </header>

      <main className="app-main">
        {status === 'idle' && (
          <div className="big-body">
            <div className="body">
              <VideoUpload
                onUploadStart={() => setStatus('uploading')}
                onJobCreated={handleJobCreated}
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
            <div className="about">
              <div>
                <h3 className="about-title">about nuu</h3>
              </div>

              <p>We've all been there before: standing in IKEA and staring at a beautiful couch, wondering, "Would this fit in my apartment? Would it even look good?" We can't always afford expensive mistakes like buying a desk that's too big, or a bookshelf that blocks the only outlet in the room. What we need is a way to carry our space with us, like digital shell we could reference anytime and anywhere.</p>

              <p>Nuu transforms any room into an interactive 3D model that you can pull up on your phone or laptop and navigate with first-person or third-person controls. Take your virtual room with you to furniture stores, compare dimensions in real-time, and visualize how new pieces will look. Or for those touring new apartments or homes, turn your videos into virtual tours you can look back to at any time.</p>
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
            <p>{processingMessage}</p>
            {jobId && <p className="status-sub">Job ID: {jobId}</p>}
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