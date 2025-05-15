import { Gauge, gaugeClasses } from '@mui/x-charts/Gauge'

import React, { useEffect, useState } from 'react'

import './FadVisualizer.css'

const FadVisualizer = ({fadScore}) => {

    // FAD Score State
    const [fadCategory, setFadCategory] = useState('Poor')

    // Create Function that returns a color based on the FAD Score
    const getFadColor = (score) => {
        if (score >= 0 && score <= 25) {
            return 'red'
        } else if (score > 25 && score <= 50) {
            return 'orange'
        } else if (score > 50 && score <= 75) {
            return 'yellow'
        } else if (score > 75 && score <= 100) {
            return 'lime'
        }
    }

    const getFadCategory = (score) => {
        if (score >= 0 && score <= 25) {
            return 'Poor'
        } else if (score > 25 && score <= 50) {
            return 'Fair'
        } else if (score > 50 && score <= 75) {
            return 'Good'
        } else if (score > 75 && score <= 100) {
            return 'Excellent'
        }
    }

    return (
        <div className='fad-visualizer-area'>
            <div className="fad-visualizer-header">
                FAD SCORE
            </div>
            <div className="fad-visualizer-body">
                <div className="gauge-area">
                    <Gauge
                        value={fadScore}
                        startAngle={-135}
                        endAngle={135}
                        innerRadius="90%"
                        outerRadius="100%"
                        sx={{
                            [`& .${gaugeClasses.valueText}`]: {
                              fontSize: 0,
                              transform: 'translate(0px, -10px)',
                            },
                            [`& .${gaugeClasses.valueArc}`]: {
                                fill: `${getFadColor(fadScore)}`, // Color of the Gauge
                            },
                        }}
                        text={
                            ({ value, valueMax }) => `${75} / ${100}`
                        }
                        // ...
                    />
                    <div className="fad-score">{fadScore}</div>
                </div>
            </div>
            <div className="fad-visualizer-footer">
                <div className="score-dot-area">
                    <div className="score-dot" style={{marginLeft: `calc(${fadScore}% - 7.5px)`, backgroundColor: `${getFadColor(fadScore)}`}}></div>
                </div>
                <div className="fad-score-bar">
                    <div className="fad-score-bar-fill" style={{width: `${fadScore}%`, backgroundColor: `${getFadColor(fadScore)}`}}></div>
                </div>
                <div className="fad-category">
                    {getFadCategory(fadScore)}
                </div>

            </div>
        </div>
    )
}

export default FadVisualizer