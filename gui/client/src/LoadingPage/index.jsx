import Loader from '../Loader'

import './LoadingPage.css' 

const LoadingPage = ({message, progress}) => {

    return (
        <div className="loading-page-area">
            <div className="loader-area">
                <Loader/>
            </div>

            <div className="message">
                {message}
            </div>

            {/* Progress Bar */}
            <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            </div>
            
        </div>
    )

}

export default LoadingPage