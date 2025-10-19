import React from 'react'
import { useState, useRef } from 'react'
import { getApiUrl } from '../config/api'
import './VideoUpload.css'

export default function VideoUpload({ 
  onUploadStart, 
  onJobCreated, 
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

      console.log('Uploading video to backend...')
      
      // Upload video to backend
      const uploadResponse = await fetch(getApiUrl('/api/upload'), {
        method: 'POST',
        body: formData,
      })

      console.log(uploadResponse)

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.status}`)
      }

      const uploadResult = await uploadResponse.json()
      console.log('Upload response:', uploadResult)
      
      // Backend returns job ID, start polling for status
      if (uploadResult.jobId) {
        onJobCreated(uploadResult.jobId)
      } else {
        throw new Error('No job ID returned from backend')
      }

    } catch (error) {
      console.error('Upload error:', error)
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        onError('Backend server is not running. Please start the backend server first.')
      } else {
        onError(`Failed to upload video: ${error.message}`)
      }
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
            <div className="upload-icon"><img src={"/nuu_transparent.png"} alt={"nuu"} height={100}/></div> 
            <p>Drag and drop your file here, or click to browse</p>
            <p className="upload-hint">
              File size limit: 100MB
              <br/>
              File types accepted: .mp4, .mov, .avi, .glb, .gltf, .obj
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
