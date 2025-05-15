import { BarChart, Bar, ResponsiveContainer, Tooltip, CartesianGrid, XAxis, YAxis, Legend } from "recharts"

import './Chart.css'
import { useEffect } from "react"

const Chart = ({type, fadScores, time}) => {

    const fad_data = [
        {model: "Our Model", fad_score: fadScores.original},
        {model: "Benchmark", fad_score: fadScores.generated}
    ]

    const time_data = [
        {model: "Our Model", time: time},
        {model: "Benchmark", time: time}   
    ]


    return (
        <>
            {!type && <ResponsiveContainer width={"100%"} height={"100%"} >
                <BarChart width={48} height={48} data={time_data}>
                    <CartesianGrid strokeDasharray="3 3"/>
                    <XAxis dataKey="model" stroke="#FFFFFF"/>
                    <YAxis stroke="#FFFFFF"/>
                    <Bar dataKey="time" fill="#D3D3D3" barSize={40}/>
                </BarChart>
            </ResponsiveContainer>}
            {type && <ResponsiveContainer width={"100%"} height={"100%"} >
                <BarChart width={48} height={48} data={fad_data}>
                    <CartesianGrid strokeDasharray="3 3"/>
                    <XAxis dataKey="model" stroke="#FFFFFF"/>
                    <YAxis stroke="#FFFFFF"/>
                    <Bar dataKey="fad_score" fill="#D3D3D3" barSize={40}/>
                </BarChart>
            </ResponsiveContainer>}
        </>
    )
}

export default Chart