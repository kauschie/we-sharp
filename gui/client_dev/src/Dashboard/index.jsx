import FadVisualizer from './FadVisualizer'
import Chart from './Chart'

import { useEffect, useState } from 'react'

import './Dashboard.css'

const Dashboard = ({closeDashboard, generationData}) => {

    const [fad, setFad] = useState(generationData.fad_scores.original)

    useEffect(() => {
        console.log(generationData.fad_scores)
    }, [])

    return (
        <div className="dashboard-area">
            <div className="dashboard-header">
                <div className="dashboard-header-left">
                    <h2 onClick={() => {setFad(generationData.fad_scores.original)}}>Our Model</h2>
                    <h2 onClick={() => {setFad(generationData.fad_scores.generated)}}>Benchmark</h2>
                </div>
                <div className="dashboard-header-right">
                    <h2 onClick={() => {closeDashboard(false)}}>Exit</h2>
                </div>
            </div>

            <div className="dashboard-body">
                <div className="dashboard-wrapper">           
                    <div className="dashboard-left">
                        <div className="fad-menu-area">
                            <FadVisualizer fadScore={fad}/>
                        </div>
                    </div>
                    <div className="dashboard-right">
                        <div className="fad-chart-area">
                            <div className="dashboard-headers">
                                <h3>FAD Score</h3>
                            </div>
                            <div className="chart-container">
                                <div className="chart-area">
                                    {<Chart type={1} fadScores={generationData.fad_scores} time={generationData.generation_time}/>}
                                </div>
                            </div>
                        </div>

                        <div className="generation-time-chart">
                            <div className="dashboard-headers">
                                <h3>Generation Time</h3>
                            </div>
                            <div className="chart-container">
                                <div className="chart-area">
                                    {<Chart type={0} fadScores={generationData.fad_scores} time={generationData.generation_time}/>}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    )
}

export default Dashboard