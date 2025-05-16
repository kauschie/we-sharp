import AudioBar from '../javascript/AudioBar'

import './Visualizer.css'

// Import the sample generation functions
import {
    generateRandomSamples,
    generatePulseSamples, 
    generateOscillatingSamples 
} from '../javascript/SampleGenerators'

import {forwardRef, useEffect, useRef, useState} from 'react'

const Visualizer = forwardRef(({...props}, ref) => {

    const [stride, setStride] = useState(0)

    // Reference Variables
    const barsRef = useRef([])
    const canvasRef = useRef()
    const patternIndexRef = useRef(0)

    const arrayRef = useRef([])

    // Microphone Ref and AudioPlayer Ref
    const micRef = ref?.micRef
    const audioPlayerRef = ref?.audioPlayerRef

    // Array of sample generation functions
    const samplePatterns = [
        (    ) => generateRandomSamples(),
        (time) => generatePulseSamples(time),
        (time) => generateOscillatingSamples(time),
        (time) => generatePulseSamples(time)
    ]

    // Functions
    const createBars = () => {
        const barCount = 512 / 2 // Half of the fftSize
    
        if (barsRef.current.length === 0) { // Prevent re-creation
            for (let i = 0; i < barCount; i++) {
                const color = `hsl(${i}, 100%, 50%)`
                barsRef.current.push(new AudioBar(color, i))
            }
        }
    }

    const createArray = (canvas) => {
        let bar_width = 5
        let array_length = canvas.width / (bar_width + 10)
        for (let i = 0; i < array_length; i++) {
            arrayRef.current.push(0)
        }
    }

    const shiftArray = (height) => {

        let arr = arrayRef.current

        // Shift array values except last value
        for (let i = 0; i < (arr.length - 1); i++) {
            arr[i] = arr[i + 1]
        }

        // Set last value to the argument
        arr[arr.length - 1] = height
    }

    const drawEqualizer = (context, canvas, samples) => {
        // Clear the canvas for upcoming draw
        context.clearRect(0, 0, context.canvas.width, context.canvas.height)
        // Update and draw each bar based on sample data
        barsRef.current.forEach((bar, i) => {
            bar.update(samples[i])
            bar.radial_visualizer(context, canvas)
        })
    }

    const drawAudioWave = (context, canvas, samples, test) => {

        let total = 0.0
        let average = 0.0

        barsRef.current.forEach((bar, i) => {
            bar.update(samples[i])
            total += bar.height
        })

        average = (total / barsRef.current.length) * 0.5

        shiftArray(average)

        // Clear the canvas for upcoming draw
        context.clearRect(0, 0, context.canvas.width, context.canvas.height)

        context.beginPath()

        arrayRef.current.forEach((element, index) => {

            let horizontal_position = (index * 10)

            context.fillStyle = 'white'
            context.fillRect(horizontal_position, canvas.height / 2, 5,  2.5 + element)
            context.fillRect(horizontal_position, canvas.height / 2, 5, -2.5 - element)
        })

        context.stroke()

    }

    const resizeCanvas = (canvas, context) => {
        const parent = props.parentref.current
        if (!parent) return

        const size = Math.min(parent.clientWidth, parent.clientHeight)
        const dpr = window.devicePixelRatio || 1

        // Reset canvas size to avoid scaling issues
        canvas.width = 0
        canvas.height = 0

        // Set CSS 
        canvas.width = `${size}px`
        canvas.height = `${size}px`

        // Set canvas dimensions based on DPR
        canvas.width = size * dpr
        canvas.height = size * dpr

        // Reset the transform to clear any previous scaling
        // context.resetTransform()

        // Scale canvas context
        // context.scale(dpr, dpr) (Causes issues with alignment)

        createArray(canvas)
        createBars()
    }

    // UseEffects

    // Change the sample pattern every 5 seconds
    useEffect(() => {
        const intervalId = setInterval(() => {
            // Cycle through patterns
            patternIndexRef.current = (patternIndexRef.current + 1) % samplePatterns.length
        }, 7500) // Switch every 5000ms (5 seconds)

        // Cleanup interval on component unmount
        return () => clearInterval(intervalId)
    }, [])

    useEffect(() => {
        // Set up the canvas
        const canvas = canvasRef.current
        const context = canvas.getContext('2d')

        // If statement if ref is null or not
        if (props.parentref) {
            // Do something with the parent ref
            resizeCanvas(canvas, context)
        }
        else {
            const dpr = window.devicePixelRatio || 1;

            // Set canvas dimensions based on DPR
            canvas.width = canvas.offsetWidth * dpr;
            canvas.height = canvas.offsetHeight * dpr;
    
            // Scale canvas context
            context.scale(dpr, dpr);
        }

        window.addEventListener("resize", resizeCanvas);
        let animationID

        // Initialize AudioBars
        createBars()

        // Initialize Array
        createArray(canvas)

        let test = 0

        // Animate Function
        const renderer = () => {

            let samples

            // When the microphone is on, render the audio data
            // Otherwise render generated audio data
            if (props.vis_type) {
                if (micRef.current && micRef.current.initialized) {
                    samples = micRef.current.getSamples()
                    drawEqualizer(context, canvas, samples)
                }
                else if (audioPlayerRef.current && audioPlayerRef.current.initialized) {
                    samples = audioPlayerRef.current.getSamples()
                    drawEqualizer(context, canvas, samples)
                }
                else {
                    const currentPattern = samplePatterns[patternIndexRef.current]
                    samples = currentPattern(Date.now() * 0.001)
                    drawEqualizer(context, canvas, samples)
                }
            }
            else {
                if (micRef.current && micRef.current.initialized) {
                    samples = micRef.current.getSamples()
                    test += 1
                    drawAudioWave(context, canvas, samples, test)
                }
            }

            // Animate the next frame
            animationID = requestAnimationFrame(renderer)

        }
        renderer()

    }, [])

    return(
        <>
            {props.vis_type
                ? <canvas className='equalizer-canvas' ref={canvasRef} {...props}/>
                : <canvas className='waveform-canvas' ref={canvasRef} {...props}/>
            }
        </>  
    )
})

export default Visualizer