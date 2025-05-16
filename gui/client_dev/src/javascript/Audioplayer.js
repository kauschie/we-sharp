class Audioplayer {
  constructor(audio_input) {
    this.initialized = false

    this.audio = new Audio(audio_input)
    this.audio.crossOrigin = "anonymous"

    // Retrieve audio properties using Web Audio API
    this.audioContext = new AudioContext()
    this.audioSource = this.audioContext.createMediaElementSource(this.audio)
    this.analyser = this.audioContext.createAnalyser()
    this.audioSource.connect(this.analyser)
    this.analyser.connect(this.audioContext.destination)
    // Must be a power of 2 (Range: 2^5 <--> 2^15)
    this.analyser.fftSize = 512
    // Convert audio data into an array
    const bufferLength = this.analyser.frequencyBinCount
    this.dataArray = new Uint8Array(bufferLength)
  }

  getSamples() {
    // Copy current waveform into this.dataArray
    this.analyser.getByteTimeDomainData(this.dataArray)
    // normalize samples to range between -1 and 1 (represents a wav)
    let normalized_samples = [...this.dataArray].map(element => (element / 128) - 1)
    return normalized_samples
  }

  play() {
    this.audio.play()
  }

  pause() {
    this.audio.pause()
  }

  remove() {
    this.audio.pause() // Pause the audio
    this.audio.currentTime = 0 // Reset the playback position
    this.audio = null // Remove the reference to the audio object
  }

  setSource(src) {
    this.audio.src = src
  }

}

export default Audioplayer