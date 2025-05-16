import React, { useState } from 'react';

import './Textbox.css'

const Textbox = () => {
    const [text, setText] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleChange = (event) => {
        setText(event.target.value);
    };

    const sendRequest = async () => {
        if (!text) {
            alert("Please enter notation before generating MIDI.");
            return;
        }

        setIsLoading(true);

        try {
            // Send request to Flask API
            const response = await fetch(`https://136.168.201.107:3000/generate_midi?notation=${text}`, {
                method: 'GET',
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            // Convert response to a blob
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);

            // Create a link element to trigger the download
            const a = document.createElement('a');
            a.href = url;
            a.download = "generated.mid";
            document.body.appendChild(a);
            a.click();

            // Cleanup
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error("Error fetching MIDI file:", error);
            alert("Failed to generate MIDI file.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className='textbox-area'>
            <input
                type="text"
                value={text}
                onChange={handleChange}
                placeholder="Enter notation (e.g., 5334221234555)"
            />
            <button onClick={sendRequest} disabled={isLoading}>
                {isLoading ? "Generating..." : "Generate MIDI"}
            </button>
        </div>
    );
};

export default Textbox;