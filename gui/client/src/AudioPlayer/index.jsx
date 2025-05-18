import React, { useState, useEffect } from 'react'
import './AudioPlayer.css'

const AudioPlayer = ({ audioPlayerRef }) => {
  const [isPlaying, setIsPlaying] = useState(false) // Play/Pause state
  const [currentTime, setCurrentTime] = useState(0) // Current playback time
  const [duration, setDuration] = useState(0) // Total duration of the audio
  const [volume, setVolume] = useState(1) // Volume state (0 to 1)
  const [isVolumeVisible, setIsVolumeVisible] = useState(false) // Volume control visibility

  // Play/Pause toggle
  const togglePlayPause = () => {
    if (audioPlayerRef.current) {
      if (isPlaying) {
        audioPlayerRef.current.initialized = false
        audioPlayerRef.current.pause()
      } else {
        audioPlayerRef.current.initialized = true
        audioPlayerRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  // Handle slider change (seeking)
  const handleSliderChange = (e) => {
    const newTime = parseFloat(e.target.value)
    if (audioPlayerRef.current) {
      audioPlayerRef.current.audio.currentTime = newTime
      setCurrentTime(newTime)
    }
  }

  // Handle volume change
  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value)
    if (audioPlayerRef.current) {
      audioPlayerRef.current.audio.volume = newVolume
      setVolume(newVolume)
    }
  }

  // Toggle volume slider visibility
  const toggleVolumeVisibility = () => {
    setIsVolumeVisible(!isVolumeVisible)
  }

  // Format time in MM:SS
  const formatTime = (time) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`
  }

  // Render volume icon based on volume level
  const renderVolumeIcon = () => {
    if (volume === 0) {
      return <img className='volume-icon' src="./graphics/muteBtn.svg" alt="Mute"/>
    } else if (volume < 0.5) {
      return <img className='volume-icon' src="./graphics/volumeDownBtn.svg" alt="Volume Down"/>
    } else {
      return <img className='volume-icon' src="./graphics/volumeUpBtn.svg" alt="Volume Up"/>
    }
  }

  // Use effect to control renderVolumeIcon
  useEffect(() => {
    renderVolumeIcon()
  }, [volume])

  // Sync playback state and duration
  useEffect(() => {
    const audioPlayer = audioPlayerRef.current

    if (audioPlayer) {
      const audioElement = audioPlayer.audio

      // Update duration when metadata is loaded
      const updateDuration = () => setDuration(audioElement.duration)

      // Update current time during playback
      const updateCurrentTime = () => setCurrentTime(audioElement.currentTime)

      // Handle audio end
      const handleAudioEnd = () => {
        audioPlayerRef.current.initialized = false
        setIsPlaying(false)
      }

      audioElement.addEventListener('loadedmetadata', updateDuration)
      audioElement.addEventListener('timeupdate', updateCurrentTime)
      audioElement.addEventListener('ended', handleAudioEnd)

      return () => {
        audioElement.removeEventListener('loadedmetadata', updateDuration)
        audioElement.removeEventListener('timeupdate', updateCurrentTime)
        audioElement.removeEventListener('ended', handleAudioEnd)
      }
    }
  }, [audioPlayerRef])

  // Calculate slider fill percentage
  const sliderFillPercentage = (currentTime / duration) * 100 || 0

  return (
    <div className="audio-player">
      {/* Play/Pause Button */}
      <button className="play-pause-btn" onClick={togglePlayPause}>
        { isPlaying ? 
          <img src="./graphics/pauseBtn.svg" alt="Pause"/> : 
          <img src="./graphics/playBtn.svg" alt="Play"/>
        }
      </button>

      {/* Slider */}
      <div className="slider-container">
        <input
          type="range"
          className="slider"
          min="0"
          max={duration || 0}
          value={currentTime || 0}
          step="0.01"
          onChange={handleSliderChange}
          style={{
            background: `linear-gradient(to right, #007bff ${sliderFillPercentage}%, #ccc ${sliderFillPercentage}%)`
          }}
        />
      </div>

      {/* Remaining Time */}
      <span className="time-display">
        {formatTime(duration - currentTime)}
      </span>

      {/* Volume Control */}
      <div className="volume-control">
        <button className="volume-toggle-btn" onClick={toggleVolumeVisibility}>
          {renderVolumeIcon()}
        </button>

        { isVolumeVisible && 
          <input
            id="volume-slider"
            type="range"
            className="volume-slider"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={handleVolumeChange}
          />
        }
      </div>
    </div>
  )
}

export default AudioPlayer

