class AudioBar {
    constructor(color, index) {
        this.width = 5        // bar width
        this.height = 10      // bar height
        this.color = color    // bar color
        this.index = index    // bar index
    }

    // Adjusts variables based on microphone data
    update(microphone_input) {
        // Update the scale (Affects bar length)
        let scale = 200
        const sound = microphone_input * scale

        // Make bar shortening smoother
        if (sound > this.height) {
            this.height = sound
        } else {
            this.height -= this.height * 0.05
        }
    }

    // Applies changes to the canvas
    radial_visualizer(context, canvas) {
        // Needed for rotation
        context.save()
        // Center of the canvas
        context.translate(canvas.width / 2, canvas.height / 2)
        // Rotate Image (64 can be changed for different effects)
        context.rotate((this.index * Math.PI) / 64)
        // Draw an individual bar
        context.fillStyle = this.color
        const radius = canvas.width / 3
        context.fillRect(0, radius, this.width, this.height)
        // Needed for rotation
        context.restore()
    }

    wave_visualizer(context, canvas, test, x) {
        let x_pos = canvas.width + (this.index * x)
        x_pos = x_pos - test
        if (x_pos < 0) {
            x_pos = x_pos + canvas.width
        }
        context.beginPath()
        context.fillStyle = this.color
        context.fillRect(x_pos, canvas.height / 2, 5, 2)
        context.fillRect(x_pos, canvas.height / 2, 5, -2)
        context.stroke()
    }
}

export default AudioBar