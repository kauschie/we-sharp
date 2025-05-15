import Visualizer from './Visualizer/index.jsx'
import Logo from './Logo/index.jsx'
import AudioPlayer from './AudioPlayer/index.jsx'
import Microphone from './javascript/Microphone'
import Audioplayer from './javascript/Audioplayer'
import LoadingPage from './LoadingPage/index.jsx'
import Dashboard from './Dashboard/index.jsx'

import Textbox from './Textbox/index.jsx'
import Keyboard from './Keyboard/index.jsx'

import Options from './Options/index.jsx'

import { useEffect, useState, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'

import './App.css'

export default function App () {

  // Global Variables
  const server_ip = 'https://athena.cs.csubak.edu/dom/api'

  // State Variables
  const [mic, setMic] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)
  const [isWavReady, setIsWavReady] = useState(false)
  const [keyboard, setKeyboard] = useState(false)
  const [userId, setUserId] = useState(null)

  // Loading State, Loading Message, and Progress
  const [progress, setProgress] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [loadingPageMessage, setLoadingPageMessage] = useState("Processing")

  // Dashboard State
  const [dashboard, setDashboard] = useState(false)

  // Header Buttons state
  const [headerBtns, setHeaderBtns] = useState(false)

  // Testing generation function
  const [generationData, setGenerationData] = useState(null)

  // Options component state
  const [showOptions, setShowOptions] = useState(false)

  // Audio File index
  const [audioIndex, setAudioIndex] = useState(0)

  // Button State Variables

  const [showGenerate, setShowGenerate] = useState(false)
  const [showArrows, setShowArrows] = useState(false)


  // Reference Variables
  const micRef = useRef()
  const audioPlayerRef = useRef()
  const uploadBtnRef = useRef(null)
  const visRef = useRef(null)

  const checkForMicrophone = async () => {
    try {
      // Request access to media devices
      const devices = await navigator.mediaDevices.enumerateDevices()
      
      // Filter devices for microphones (audio inputs)
      const microphones = devices.filter(device => device.kind === 'audioinput')
      
      // Check if any microphones are available
      if (microphones.length > 0) {
        console.log("Microphone(s) available:", microphones)
        return true
      } else {
        console.log("No microphones available")
        return false;
      }
    } catch (error) {
      console.error("Error checking for microphones:", error)
      return false
    }
  }

  // Function that toggles the state of the microphone (on/off)
  const toggleRecording = async () => {
    // Turn off options
    setShowOptions(false)
    // Check if the microphone is already initialized
    if (!micRef.current) {
      // Check if the browser supports the microphone
      const hasMic = await checkForMicrophone()
      // Check if the browser supports the microphone
      if (!hasMic) {
        alert("No microphone available")
        return
      }
      // Remove the audio player component and other ui components if displayed
      setIsWavReady(false)
      setShowGenerate(false)
      setShowArrows(false)
      // Turn off the audio player if initialized
      if (audioPlayerRef.current) {
        audioPlayerRef.current.initialized = false
        audioPlayerRef.current.pause()
      }
      // Initialize the microphone
      micRef.current = new Microphone()
      // Begin recording (Wait for initialization)
      await micRef.current.ready
      await micRef.current.startRecording()
      // Update Mic State
      setMic(true)
    } 
    else {
      // Record Audio and store on Athena 
      try {
        // Begin Loading State
        setLoadingPageMessage("Uploading Audio")
        setIsLoading(true)

        // Stop recording and get the audio blob
        const audioBlob = await micRef.current.stopRecording()
        const formData = new FormData()
        formData.append('file', audioBlob, 'uploaded.wav')
        // Append User ID to the form data
        // console.log("Verifying User ID:", localStorage.getItem('user_id'))
        // formData.append('user_id', localStorage.getItem('user_id'))
        const user_id = localStorage.getItem('user_id')
      
        // Send audio to backend
        await fetch(`${server_ip}/upload`, {
          method: 'POST',
          headers: {
            'user_id': user_id,  // Add user_id in a custom header
          },
          body: formData,
        })
        .then((res) => {
          if (res.ok) {
            console.log("File Uploaded successfully")
            // Update Loading Message
            setLoadingPageMessage("Setting up Audio Player")
            // Initialize the audio player
            initAudioPlayer("uploaded", false)
            // Turn off the loading state
            setIsLoading(false)
          }
        })
      } catch (e) {
        console.error(e)
      }

      // Turn off the microphone
      micRef.current.stop()
      // Reset to allow re-initialization
      micRef.current = null
      // Update Mic State 
      setMic(false)
    }
  }

  const initAudioPlayer = async (folder, generateStatus = false, index = 0) => {
    let url
    let user_id = localStorage.getItem('user_id')
  
    if (folder === "uploaded") {
      url = `${server_ip}/audio/${folder}/${user_id}_${folder}.wav?t=${Date.now()}`
    }
    else if (folder === "generated") {
      url = `${server_ip}/audio/${folder}/${user_id}/audio-${index}.wav?t=${Date.now()}`
    }
    else {
      console.error("Invalid folder")
      return
    }
  
    try {
      const response = await fetch(url)
      if (!response.ok) throw new Error('Audio file not found')
      await response.blob()
    } catch (err) {
      console.error('Error loading audio:', err)
      alert('Failed to load audio. Please try again.')
      return
    }
  
    if (audioPlayerRef.current) {
      audioPlayerRef.current.remove()
    }
  
    audioPlayerRef.current = new Audioplayer(url)
    setAudioUrl(url)

    console.log("Audio URL:", url)

    // Determine if the generate button should be shown
    if (folder === "uploaded") {
      setShowGenerate(true)
    }
    else {
      setShowGenerate(false)
    }

    setIsWavReady(true)
  
    // Get the count of audio files
    const count = await getAudioFileCount()

    console.log("Generate Status:", generateStatus)
  
    if (count > 1 && generateStatus) {
      setShowArrows(true)
    } else {
      console.log("Only one audio file found or generation has not occured.")
      setAudioIndex(0)
      setShowArrows(false)
    }
  }  

  // Function to trigger file input click
  const uploadButtonClick = () => {
    // Turn off options
    setShowOptions(false)

    if (uploadBtnRef.current) {
      // Clear previous file selection so onChange always fires
      uploadBtnRef.current.value = null
      uploadBtnRef.current.click()
    }
  }

  const generateButtonClick = async () => {
    setLoadingPageMessage("Generating Audio")
    setIsLoading(true)

    const userId = localStorage.getItem("user_id")

    try {
      console.log("Starting audio generation...")
      const result = await generateAudio()
      console.log("Audio generation complete!")
    } catch (error) {
      console.error("Error during audio generation:", error)
    }

    setLoadingPageMessage("Initializing the Audio Player")
    initAudioPlayer("generated", true)
    setHeaderBtns(true)
    setIsLoading(false)
  }

  // Handle File Upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0]

    // Exit early if no file is selected
    if (!file) {
      console.log("No file selected, skipping upload.")
      return
    }

    // Set loading state and message
    setLoadingPageMessage("Uploading Audio")
    setIsLoading(true)

    // Check if the file is an audio file
    if (file && file.type.startsWith('audio/')) {
      const formData = new FormData()
      formData.append('file', file, 'uploaded.wav')
      
      // Get UUID
      const user_id = localStorage.getItem('user_id')

      try {
        const response = await fetch(`${server_ip}/upload`, {
          method: 'POST',
          headers: {
            'user_id': user_id,
          },
          body: formData,
        })

        if (response.ok) {
          console.log("Audio file uploaded successfully!")
          // Initialize the audio player
          initAudioPlayer("uploaded", false)

          // Turn off the loading state
          setIsLoading(false)
        } else {
          console.error("Error uploading file")
        }
      } catch (error) {
        console.error("Error:", error)
      }
    } else {
      alert("Please upload a valid audio file.")
    }
  }


  const generateAudio = async () => {
    try {
      const userId = localStorage.getItem("user_id")
      const filename = `${userId}_uploaded.wav`

      // Check if file exists first
  
      const response = await fetch(`${server_ip}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "user_id": userId
        },
      })
  
      if (!response.ok) {
        throw new Error(`Error processing audio: ${response.statusText}`)
      }

      const data = await response.json()
      console.log("Response data:", data)  // Log the whole data to check its structure
  
      if (data && data.fad_scores) {
        console.log(data.fad_scores.original) // Access original score
        setGenerationData(data)
        setShowOptions(true)
      } else {
        console.error("FAD scores not found in response")
      }

    } catch (error) {
      console.error("Failed to process audio:", error)
    }
  }

  // Function to close the dashboard overlay
  const closeDashboard = () => {
    setDashboard(false)
  }

  const downloadAudio = async () => {
    if (!audioUrl) {
      alert("No audio file available for download.")
      return
    }
  
    try {
      const response = await fetch(audioUrl)
      if (!response.ok) {
        throw new Error(`Failed to fetch audio file: ${response.statusText}`)
      }
  
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
  
      // Create a download link and trigger it
      const a = document.createElement("a")
      a.href = url
      a.download = `audio-${Date.now()}.wav` // Assigns a default name
      document.body.appendChild(a)
      a.click()
  
      // Clean up
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error("Error downloading audio:", error)
    }
  }

  // Function to determine how many audio files are in the directory
  const getAudioFileCount = async () => {
    const userId = localStorage.getItem("user_id")
    try {
      const response = await fetch(`${server_ip}/audio_count/`, {
        headers: {
          "user_id": userId,
        },
      })
      if (!response.ok) {
        throw new Error(`Failed to fetch audio file count: ${response.statusText}`)
      }
      const data = await response.json()
      const count = data.count

      console.log("Number of audio files:", count)
      return count
    } catch (error) {
      console.error("Error fetching audio file count:", error)
      return 0
    }
  }
  
  // On page load, assign user a uuid
  useEffect(() => {
    let userId = localStorage.getItem("user_id")
    
    if (!userId) {
      userId = uuidv4()
      localStorage.setItem("user_id", userId)
      console.log("User ID assigned:", userId)
    } else {
      console.log("Using Existing User ID:", userId)
    }

    setUserId(userId)
  
    // Initialize user directory on backend
    fetch(`${server_ip}/init-user`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "user_id": userId,
      },
    })
      .then((res) => res.json())
      .then((data) => console.log("Init User Directory:", data.message))
      .catch((err) => console.error("Failed to init user directory:", err))
  }, [])
  
  // Polling for progress
  useEffect(() => {
    if (!isLoading) return
  
    const userId = localStorage.getItem("user_id")
  
    console.log("Starting progress polling...")
  
    const interval = setInterval(async () => {
      console.log("Polling progress...")
  
      try {
        const response = await fetch(`${server_ip}/progress`, {
          headers: { "user_id": userId },
        })
  
        if (response.ok) {
          const data = await response.json()

          let message = `Generating ${data.stage} | Time Remaining: ${data.eta_seconds} | Current Length: ${data.length}`

          setLoadingPageMessage(message)
          setProgress(data.progress)
        }
      } catch (error) {
        console.error("Error fetching progress:", error)
      }
    }, 1000) // Poll every second
  
    // Cleanup function to stop polling when component unmounts or `isLoading` changes
    return () => {
      console.log("Stopping progress polling...")
      clearInterval(interval)
    }
  }, [isLoading]) // Only runs when `isLoading` changes
  
  return (
    <div className='app-body'>

      <Textbox/>
      { keyboard && <Keyboard setKeyboard={setKeyboard} userId={userId} setIsLoading={setIsLoading} setMessage={setLoadingPageMessage} initAudioPlayer={initAudioPlayer} setShowOptions={setShowOptions} setGenerate={setShowGenerate} />}

      { isLoading && <LoadingPage message={loadingPageMessage} progress={progress}/> }
      { dashboard && <Dashboard closeDashboard={setDashboard} generationData={generationData}/> }

      <div className="header">
          <Logo/>
      </div>

      <Options showAll={showOptions} download={downloadAudio} setDashboard={setDashboard} setKeyboard={setKeyboard}/>

      <div className="primary-canvas-area" ref={visRef}>
        {showArrows && <button
          className='button-85'
          onClick={async () => {
            const count = await getAudioFileCount()
            if (count === 0) return

            const nextIndex = (audioIndex - 1 + count) % count
            setAudioIndex(nextIndex)
            initAudioPlayer("generated", true, nextIndex)
          }}
        >
          {`< ${audioIndex}`}
        </button>}

        <Visualizer ref={{ micRef: micRef, audioPlayerRef: audioPlayerRef }} vis_type={1} parentref={visRef} />

        {showArrows && <button
          className='button-85'
          onClick={async () => {
            const count = await getAudioFileCount()
            if (count === 0) return

            const nextIndex = (audioIndex + 1) % count
            setAudioIndex(nextIndex)
            initAudioPlayer("generated", true, nextIndex)
          }}
        >
          {`${audioIndex} >`}
        </button>}
      </div>

      <div className="secondary-canvas-area">
        { isWavReady && audioUrl && (<AudioPlayer key={audioUrl} audioUrl={audioUrl} audioPlayerRef={audioPlayerRef} />) }
      </div>

      <div className="footer">
        <button className='button-85' onClick={() => {toggleRecording()}}>
          {mic ? 'Stop' : 'Record'}
        </button>

        { showGenerate && <button className='button-85' onClick={generateButtonClick}>Generate</button> }

        {/* File Upload Button */}
        <input className="file-input"
          type="file" 
          accept="audio/*"
          ref={uploadBtnRef}
          style={{display: 'none'}}
          onChange={handleFileUpload}
        />

        <button className='button-85' onClick={uploadButtonClick}>
          Upload
        </button>

      </div>

    </div>
  )
}