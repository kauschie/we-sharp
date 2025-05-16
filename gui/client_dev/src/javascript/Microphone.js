class Microphone {
    constructor() {
        this.initialized = false
        // For Audio Stream
        this.stream = null
        // For Audio Recording
        this.mediaRecorder = null
        this.audioChunks = []

        // Initialize Promise for Audio Stream Setup
        this.ready = navigator.mediaDevices.getUserMedia({ audio:true })
        .then((stream) => {
            // Retrieve audio properties using Web Audio API
            this.audioContext = new AudioContext()
            this.stream = stream 
            // Converts raw audio data into audio nodes
            this.microphone = this.audioContext.createMediaStreamSource(stream)
            // Processes data for visualization
            this.analyser = this.audioContext.createAnalyser()
            // Must be a power of 2 (Range: 2^5 <--> 2^15)
            this.analyser.fftSize = 512
            // Convert audio data into an array
            const bufferLength = this.analyser.frequencyBinCount
            this.dataArray = new Uint8Array(bufferLength)
            // Direct data between audio nodes
            this.microphone.connect(this.analyser)

            // Set up Media Recorder
            this.mediaRecorder = new MediaRecorder(stream)
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data)
                }
            }

            this.initialized = true
        })
        .catch(this.handleError.bind(this))
    }

    // Audio Recording
    async startRecording() {
        // Ensure that the stream and MediaRecorder are initialized
        await this.ready
        if (this.mediaRecorder && this.mediaRecorder.state === 'inactive') {
            this.audioChunks = [] // Clear previous recordings
            this.mediaRecorder.start()
            console.log('Recording started')
        }
    }

    // Stops recording audio and prepares it for download
    stopRecording() {
        return new Promise((resolve, reject) => {
            if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop()
                console.log('Recording stopped')
                this.mediaRecorder.onstop = async () => {
                    try {
                        const audioBlob = await this.createWavBlob(this.audioChunks)
                        resolve(audioBlob)
                    } catch (e) {
                        reject(e)
                    }
                }
                this.mediaRecorder.onerror = reject
            } else {
                reject('No active recording found')
            }
        })
    }

    async createWavBlob(chunks) {
        // Combine all audio chunks into a single Blob
        const webmBlob = new Blob(chunks, { type: 'audio/webm' })

        // Decode the WebM audio into PCM data
        const arrayBuffer = await webmBlob.arrayBuffer()
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer)

        // Convert PCM data to WAV format
        const wavArrayBuffer = this.audioBufferToWav(audioBuffer)
        return new Blob([wavArrayBuffer], { type: 'audio/wav' })
    }

    audioBufferToWav(audioBuffer) {
        const numberOfChannels = audioBuffer.numberOfChannels
        const sampleRate = audioBuffer.sampleRate
        const format = 1 // PCM format

        const samples = audioBuffer.getChannelData(0)
        const dataSize = samples.length * 2 // 16-bit audio
        const buffer = new ArrayBuffer(44 + dataSize)
        const view = new DataView(buffer)

        // RIFF Header
        this.writeString(view, 0, 'RIFF')
        view.setUint32(4, 36 + dataSize, true)
        this.writeString(view, 8, 'WAVE')

        // fmt subchunk
        this.writeString(view, 12, 'fmt ')
        view.setUint32(16, 16, true) // Subchunk1Size (16 for PCM)
        view.setUint16(20, format, true) // Audio format (1 = PCM)
        view.setUint16(22, numberOfChannels, true)
        view.setUint32(24, sampleRate, true)
        view.setUint32(28, sampleRate * numberOfChannels * 2, true) // Byte rate
        view.setUint16(32, numberOfChannels * 2, true) // Block align
        view.setUint16(34, 16, true) // Bits per sample

        // data subchunk
        this.writeString(view, 36, 'data')
        view.setUint32(40, dataSize, true)

        // Write PCM data
        let offset = 44
        for (let i = 0; i < samples.length; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i])) // Clamp value
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true)
            offset += 2
        }

        return buffer
    }

    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // Audio Sampling

    getSamples() {
        // Copy current waveform into this.dataArray
        this.analyser.getByteTimeDomainData(this.dataArray)
        // normalize samples to range between -1 and 1 (represents a wav)
        let normalized_samples = [...this.dataArray].map(element => (element / 128) - 1)
        return normalized_samples
    }

    getVolume() {
        // Copy current waveform into this.dataArray
        this.analyser.getByteTimeDomainData(this.dataArray)
        // normalize samples to range between -1 and 1 (represents a wav)
        let normalized_samples = [...this.dataArray].map(element => (element / 128) - 1)
        let sum = 0
        // Root Mean Square (Measure of Magnitude)
        for (let i = 0; i < normalized_samples.length; i++) {
            sum += normalized_samples[i] * normalized_samples[i]
        }
        // Get the volume
        let volume = Math.sqrt(sum / normalized_samples.length)
        return volume
    }

    stop() {
        if (this.stream) {
            // Stop all tracks in the media stream
            this.stream.getTracks().forEach(track => track.stop())
            //this.audioContext.close() // Optionally close the AudioContext
            console.log("Microphone stopped")
            this.initialized = false
        }
    }

    // Define the error handler method
    handleError(err) {
        alert(err)
    }

}

export default Microphone