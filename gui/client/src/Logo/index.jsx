import Loader from '../Loader'

import './Logo.css'

export default function Logo () {
    return(
        <div className='logo-area'>
            <div className='title'>WE-SHARP</div>
            <div className='logo'>
                <div className="logo-size">
                    <Loader/>
                </div>
            </div>
        </div>
    )
}