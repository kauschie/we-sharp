// Imports
const express = require('express')
const multer = require('multer')
const fs = require('fs')
const path = require('path')
const { spawn } = require("child_process")

// Python virtual environment path
const PYTHON_PATH_AUDIO_LM = "/home/dflores2/miniconda3/envs/audiolm_env/bin/python"
const PYTHON_PATH_FAD = "/home/dflores2/miniconda3/envs/fad/bin/python"

// Express app
const app = express()
app.use(express.json())

// Midldleware
const cors = require('cors')

app.use(cors({
  origin: 'https://athena.cs.csubak.edu/dom/api', // Allow specific origin
  methods: ['GET', 'POST'], // Allow specific methods (GET for audio files)
  allowedHeaders: ['Content-Type', 'user_id'], // Allow custom headers, including user_id
}))

// Multer setup
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "../audio/uploaded") // Save in "audio/uploaded" folder
  },
  filename: (req, file, cb) => {
    const userId = req.headers['user_id']  // Retrieve user_id from headers
    const uniqueName = `${userId}_${file.originalname}` // Include userId in the filename
    cb(null, uniqueName) // Save the file with a unique name
  },
})

const upload = multer({ storage })

// Endpoint to upload audio file
app.post('/upload', upload.single('file'), (req, res) => {
  res.status(200).send('File uploaded successfully')
})

// Endpoint to retrieve audio file
app.use('/audio/uploaded', (req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', 'https://athena.cs.csubak.edu/dom/api')
  res.setHeader('Access-Control-Allow-Methods', 'GET')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  next()
}, express.static(path.join(__dirname, '../audio/uploaded')))

// Serve static files from "audio/generated"
app.use('/audio/generated', express.static(path.join(__dirname, '../audio/generated')))

// ================================
//        Polling endpoint
// ================================

// Updated endpoint to handle progress updates using a JSON file
app.get("/progress", (req, res) => {
  const userId = req.headers["user_id"]
  if (!userId) {
    return res.status(400).json({ error: "Missing user_id" })
  }

  const progressFile = path.join(__dirname, "../audio/generated", userId, "progress.json")

  fs.readFile(progressFile, "utf8", (err, data) => {
    if (err) {
      if (err.code === "ENOENT") {
        // File not found, safe to continue with default values
        console.log(`Progress file not found for user ${userId}, returning 0% progress.`)
        return res.json({
          progress: 0,
          stage: "",
          eta_seconds: "Calculating...",
          length: 0,
          batch: null
        })
      }

      console.error("Failed to read progress file:", err.message)
      return res.status(500).json({ error: "Could not read progress file" })
    }

    try {
      const progressData = JSON.parse(data)
      const percent = parseFloat(progressData.percent)

      if (isNaN(percent)) {
        return res.status(500).json({ error: "Invalid progress value in JSON" })
      }

      console.log(`User ${userId} progress: ${percent}%`)

      // Response
      res.json({
        progress: percent,
        stage: progressData.stage,
        eta_seconds: progressData.eta_seconds,
        length: progressData.length,
        batch: progressData.batch
      })
      

    } catch (parseErr) {
      console.error("Failed to parse progress JSON:", parseErr.message)
      res.status(500).json({ error: "Malformed progress.json file" })
    }
  })
})

// Endpoint to delete progress file
app.delete("/delete_progress", (req, res) => {
  const userId = req.headers["user_id"]
  if (!userId) {
    return res.status(400).json({ error: "Missing user_id" })
  }

  const progressFile = path.join(__dirname, "../audio/generated", userId, "progress.json")

  fs.unlink(progressFile, (err) => {
    if (err) {
      if (err.code === "ENOENT") {
        return res.status(404).json({ error: "Progress file not found" })
      }
      console.error("Failed to delete progress file:", err.message)
      return res.status(500).json({ error: "Failed to delete progress file" })
    }

    console.log(`Progress file deleted for user ${userId}`)
    res.status(200).json({ message: "Progress file deleted successfully" })
  })
})


// ================================
//        UUID endpoint
// ================================

// Endpoint to initialize user directory
app.post("/init-user", (req, res) => {
  const userId = req.headers["user_id"]

  if (!userId) {
    return res.status(400).json({ error: "Missing user_id" })
  }

  const outputDir = path.join(__dirname, "../audio/generated", userId)

  try {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true })
      console.log(`Directory created for user: ${userId}`)
      return res.status(201).json({ message: "Directory created" })
    } else {
      console.log(`Directory already exists for user: ${userId}`)
      return res.status(200).json({ message: "Directory already exists" })
    }
  } catch (err) {
    console.error("Error creating directory:", err)
    return res.status(500).json({ error: "Failed to create directory" })
  }
})


// ===============================
//        FAD endpoint
// ===============================

// Function to run FAD
const runFAD = (userID) => {
  return new Promise((resolve, reject) => {
    const basePath = path.join(__dirname, `../audio/generated/${userID}`)
    const fadProcess = spawn(PYTHON_PATH_FAD, ["../scripts/fad.py", "--baseline", basePath])

    let fadScriptOutput = ""

    fadProcess.stdout.on("data", (data) => {
      fadScriptOutput += data.toString()
      console.log(`Python Output: ${data}`)
    })

    fadProcess.stderr.on("data", (data) => {
      console.error(`Python Error: ${data}`)
    })

    fadProcess.on("close", (code) => {
      if (code === 0) {
        try {
          const [fad1, fad2, fad3] = fadScriptOutput.trim().split("\n").map(Number)
          console.log(fad3, fad3)
          resolve([fad3, fad3]) // Return FAD scores as a resolved Promise
        } catch (error) {
          reject({ error: "Failed to parse FAD scores", details: error.message })
        }
      } else {
        reject({ error: "FAD script execution failed" })
      }
    })
  })
}


// ===============================
//        AudioLM endpoint
// ===============================

// Endpoint to process audio
app.post("/generate", async (req, res) => {
  const userId = req.headers["user_id"]

  if (!userId) {
    return res.status(400).json({ error: "Missing user_id" })
  }

  // Define input path from the uploaded folder
  const inputPath = path.join(__dirname, "../audio/uploaded", `${userId}_uploaded.wav`)

  // Define output path for generated audio
  const outputDir = path.join(__dirname, "../audio/generated", userId)
  const outputPath = path.join(outputDir, "audio")

  console.log("Running generate_audio.py with input:", inputPath, "and output path:", outputPath)

  // Start timing
  const startTime = Date.now()

  // Name of the script to run
  const scriptName = "gen_audio_batch2.py"

  // Spawn the Python process to generate audio (update arguments as needed)
  tags = ["--duration", "--prime_wave", "--batch_size", "--output"]
  const generateProcess = spawn(
    PYTHON_PATH_AUDIO_LM,
    [
      `../scripts/${scriptName}`,
      tags[0], 10,
      tags[1], inputPath,
      tags[2], 8,
      tags[3], outputPath
    ]
  )

  let scriptOutput = ""

  generateProcess.stdout.on("data", (data) => {
    scriptOutput += data.toString()
    console.log(`Python Output: ${data}`)
  })

  generateProcess.stderr.on("data", (data) => {
    console.error(`Python Error: ${data}`)
  })

  generateProcess.on("close", async (code) => {
    const generationTime = (Date.now() - startTime) / 1000 // seconds

    console.log("Generation time:", generationTime, "seconds")

    if (code === 0) {
      try {
        //[fadScore1, fadScore2] = await runFAD(userId)
        [fadScore1, fadScore2] = [0.5, 0.6] // Placeholder for FAD scores
        // Return a URL pointing to the new file location. Make sure your static serving routes are updated if needed.
        res.status(200).json({
          message: "Audio processed successfully",
          output: `/audio/generated/${userId}/audio.wav`,
          fad_scores: { original: fadScore1, generated: fadScore2 },
          generation_time: generationTime,
        })
      } catch (error) {
        res.status(500).json({
          error: "Failed to return audio and FAD scores",
          details: error.message,
        })
      }
    } else {
      res.status(500).json({ error: "Audio processing failed" })
    }
  })
})

// ===============================
//        Keyboard endpoint
// ===============================

// More multer setup

const keyboard_storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const userId = req.headers['user_id']  // Retrieve user_id from headers
    const userDir = path.join("../audio/generated", userId) // Directory named after user_id

    // Ensure the directory exists
    if (fs.existsSync(userDir)) {
      // Delete all files in the directory
      fs.readdirSync(userDir).forEach((file) => {
      const filePath = path.join(userDir, file)
      if (fs.lstatSync(filePath).isFile()) {
        fs.unlinkSync(filePath)
      }
      })
      // Log it
      console.log(`Deleted all files in directory: ${userDir}`)
    } else {
      fs.mkdirSync(userDir, { recursive: true })
    }

    cb(null, userDir) // Save in "audio/generated/UUID" folder
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname) // Save the file with the original name
  },
})

const keyboard_upload = multer({ storage: keyboard_storage })

app.post('/store_key_wav', keyboard_upload.single('file'), (req, res) => {
  if (!req.file) {
    console.error("No file received in the request!")
    return res.status(400).send("No file uploaded.")
  }

  console.log("Received file:", req.file)

  res.status(200).send('File uploaded successfully')
})


// ================================
//        Test if file exists
// ================================

app.get('/check-file', (req, res) => {
  const filename = req.query.filename; // e.g., "test.wav"
  const filePath = path.join(__dirname, 'public', filename); // adjust path if needed

  fs.access(filePath, fs.constants.F_OK, (err) => {
    if (err) {
      return res.json({ exists: false });
    }
    return res.json({ exists: true });
  });
})

// ===============================
//        Return Audio Count
// ===============================
// Check in a directory for the number of files with .wav at the end
app.get('/audio_count', (req, res) => {
  const user_id = req.headers['user_id']
  if (!user_id) {
    return res.status(400).json({ error: "Missing user_id" })
  }

  const dirPath = path.join(__dirname, `../audio/generated/${user_id}`)

  fs.readdir(dirPath, (err, files) => {
    if (err) {
      console.error("Error reading directory:", err);
      return res.status(500).json({ error: "Failed to read directory" })
    }

    // Filter for .wav files
    const wavFiles = files.filter(file => file.endsWith('.wav'))
    const wavCount = wavFiles.length

    res.json({ count: wavCount })
  })
})

// ===============================
//        Last endpoints
// ===============================

// Basic endpoint to test server
app.get('/', (req, res) => {
  res.send('Hello from Express!')
})

// Listen on port 3001
const PORT = 3001
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`)
})