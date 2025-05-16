// Functions to generate fake audio samples

// Generates random samples
export const generateRandomSamples = () => {
    const sampleCount = 256 // Number of samples to match the real data
    const samples = []

    // Generate random samples between -1 and 1
    for (let i = 0; i < sampleCount; i++) {
        const randomValue = Math.random() * 2 - 1.9 // Range from -1 to 1
        samples.push(randomValue)
    }

    return samples
}

// Generates pulse-like samples
export const generatePulseSamples = (time) => {
    const sampleCount = 256
    const samples = []
    const pulseSpeed = 2 // Controls the speed of the pulse
    const amplitude = Math.sin(time * pulseSpeed) * 0.075 // Pulsating amplitude

    for (let i = 0; i < sampleCount; i++) {
        const value = amplitude * Math.sin((i / sampleCount) * 2) * (Math.random() * 2 - 1)
        samples.push(value)
    }

    return samples
}

// Generates oscillating wave samples
export const generateOscillatingSamples = (time) => {
    const sampleCount = 256
    const samples = []

    for (let i = 0; i < sampleCount; i++) {
        // A combination of sine waves for a complex oscillating pattern
        const value = 0.05 * (Math.sin((i/100)) + Math.sin((i / 20) - time)) * (Math.random() * 2 - 1)
        samples.push(value)
    }

    return samples
}

// More samples to be added here