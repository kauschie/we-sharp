import './Keyboard.css'
import { useState } from 'react'

const Keyboard = ({ setKeyboard, userId, setIsLoading, setMessage, initAudioPlayer, setShowOptions, setGenerate }) => {
    const [notes, setNotes] = useState("")
    const [sequence, setSequence] = useState("")
    const [lastTimestamp, setLastTimestamp] = useState(null)

    const [firstTimestamp, setFirstTimestamp] = useState(null)

    // Mapping of keys to numbers
    const keyMap = {
        "C": "1",
        "D": "2",
        "E": "3",
        "F": "4",
        "G": "5",
        "A": "6",
        "B": "7"
    }

    const handleKeyPress = (note) => {
        // Play the note sound
        const audio = new Audio(`./sounds/${note}4.mp3`)
        audio.play()

        // Get the current timestamp
        const timestamp = Date.now()
        const num = keyMap[note]

        let timeDiff = 0

        if (lastTimestamp === null) {
            setFirstTimestamp(timestamp)
            timeDiff = 0
        }
        else {
            timeDiff = timestamp - firstTimestamp
        }

        // Append the new entry to the sequence
        setNotes(prev => prev + note)
        setSequence(prev => prev + `${num}[${timeDiff}]`)
        setLastTimestamp(timestamp)
    }

    const resetSequence = () => {
        setNotes("") // Clear the displayed notes
        setSequence("") // Clear the sequence
        setLastTimestamp(null) // Reset time tracking
        setFirstTimestamp(null) // Reset first timestamp
    }

    const generateMusic = () => {
        if (sequence === "") {
            alert("Please play some notes first")
            return
        }

        // Activate the loading state
        setMessage("Generating music...")
        setIsLoading(true)

        console.log("Generated music using sequence: ", sequence)

        // Fetch request to the server to generate the WAV file
        fetch(`https://athena.cs.csubak.edu/bra/flask/generate_wav?notation=${sequence}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok")
            }
            return response.blob() // Expecting a WAV file as a blob
        })
        .then(blob => {
            // Create a FormData object to send the file to the express server
            const formData = new FormData()
            const fileName = `audio-0.wav`
            formData.append('file', blob, fileName)

            // Send the WAV file to the express server for storage
            return fetch('https://athena.cs.csubak.edu/dom/api/store_key_wav', {
                method: 'POST',
                body: formData,
                headers: {
                    'user_id': userId // Pass the user ID through the header
                },
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to store the WAV file on the server")
            }
            console.log(response)
            console.log("WAV file stored successfully on the server")

            // Initialize the audio player
            initAudioPlayer("generated", true)

            // Deactivate the loading state and close keyboard
            setShowOptions(true)
            setIsLoading(false)
            setKeyboard(false)
        })
        .catch(error => {
            console.error("Error:", error)
        })
    }

    return (
        <div className="background-area">
            <button className="exit-button" onClick={() => setKeyboard(false)}>Exit</button>
            <div className="overlay-area">
                <div className="keyboard">
                    {Object.keys(keyMap).map(note => (
                        <div key={note} className="key" onClick={() => handleKeyPress(note)}>
                            {note}
                        </div>
                    ))}
                </div>
                <div className="buttons-area">
                    <button className="button-rgb" onClick={resetSequence}>Restart</button>
                    <button className="button-rgb" onClick={generateMusic}>Generate</button>
                </div>
                {notes && (
                    <div className="sequence-area">
                        <p>{notes}</p>
                    </div>
                )}
            </div>
        </div>
    )
}

export default Keyboard