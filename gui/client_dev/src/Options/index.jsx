import './Options.css'

const Options = ({showAll, download, setDashboard, setKeyboard}) => {

    return (
        <div className="options-area">
            {showAll && (
                <>
                    {/* Download Button */}
                    <button onClick={() => download()}>
                        <img src="./graphics/download.png" alt="Download"/>
                    </button>
                    {/* Display Dashboard
                    <button onClick={() => setDashboard(true)}>
                        <img src="./graphics/statistics.png" alt="Download"/>
                    </button>
                    */}
                </>
            )}
                    
                {/* Display Keyboard */}
                <button onClick={() => setKeyboard(true)}>
                    <img src="./graphics/music.png" alt="Download"/>
                </button>
        </div>
    )
}

export default Options