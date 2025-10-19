import React from 'react'
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
    // Check if it's a video file
    if (file.type.startsWith('video/')) {
      if (file.size > 100 * 1024 * 1024) { // 100MB limit
        onError('Video file is too large. Please select a file under 100MB.')
        return
      }
    } 
    // Check if it's a 3D model file
    else if (file.name.toLowerCase().endsWith('.glb') || file.name.toLowerCase().endsWith('.gltf') || file.name.toLowerCase().endsWith('.obj')) {
      // Handle 3D model file directly
      const modelUrl = URL.createObjectURL(file)
      console.log('3D Model file detected:', file.name, 'Type:', file.type)
      console.log('Generated URL:', modelUrl)
      onComplete(modelUrl)
      return
    }
    else {
      onError('Please select a video file or 3D model (.glb, .gltf, .obj)')
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
      <div className="upload-container">
        <div
          className={`upload-area ${isDragOver ? 'drag-over' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={handleClick}
        >
          <div className="upload-content">
            <div className="upload-icon"><img src={"../assets/nuu.JPG"} alt={"nuu"}/></div> 
            <h3>Upload Room Video or 3D Model</h3>
            <p>Drag and drop your file here, or click to browse</p>
            <p className="upload-hint">
              Video: MP4, MOV, AVI • 3D Models: GLB, GLTF, OBJ
            </p>
          </div>
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,.glb,.gltf,.obj"
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />
      </div>
    </div>
  )
}
