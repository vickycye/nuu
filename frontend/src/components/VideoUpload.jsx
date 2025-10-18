import { useState, useRef } from 'react'
import './VideoUpload.css'

export default function VideoUpload({ 
  onUploadStart, 
  onProcessingStart, 
  onComplete, 
  onError 
}) {
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileSelect = async (file) => {
    if (!file.type.startsWith('video/')) {
      onError('Please select a video file')
      return
    }

    if (file.size > 100 * 1024 * 1024) { // 100MB limit
      onError('Video file is too large. Please select a file under 100MB.')
      return
    }

    onUploadStart()

    try {
      const formData = new FormData()
      formData.append('video', file)

      // Upload video
      const uploadResponse = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!uploadResponse.ok) {
        throw new Error('Upload failed')
      }

      const uploadResult = await uploadResponse.json()
      onProcessingStart()

      // Poll for processing completion
      const pollForCompletion = async () => {
        try {
          const statusResponse = await fetch(`/api/status/${uploadResult.jobId}`)
          const status = await statusResponse.json()

          if (status.status === 'complete') {
            onComplete(status.modelUrl)
          } else if (status.status === 'error') {
            onError(status.error || 'Processing failed')
          } else {
            // Still processing, poll again in 2 seconds
            setTimeout(pollForCompletion, 2000)
          }
        } catch (error) {
          onError('Failed to check processing status')
        }
      }

      pollForCompletion()

    } catch (error) {
      onError('Failed to upload video. Please try again.')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleFileInputChange = (e) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="video-upload">
      <div
        className={`upload-area ${isDragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <div className="upload-content">
          <div className="upload-icon">📹</div>
          <h3>Upload Room Video</h3>
          <p>Drag and drop your video here, or click to browse</p>
          <p className="upload-hint">
            Supported formats: MP4, MOV, AVI • Max size: 100MB
          </p>
        </div>
      </div>
      
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />
      
      <div className="upload-tips">
        <h4>📱 Tips for best results:</h4>
        <ul>
          <li>Walk slowly around the room</li>
          <li>Keep the camera steady</li>
          <li>Make sure good lighting</li>
          <li>Include walls, floor, and furniture</li>
        </ul>
      </div>
    </div>
  )
}
