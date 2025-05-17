import subprocess
import os

# Define the absolute or relative paths to your scripts
scripts = [
    r"stan_MIDI.py",
    r"full_midi_pipeline.py",
    r"Midi_Tokenform.py",
    r"LLMshit\Yuliao\ZhengLi.py",
    r"LLMshit\Training\main.py"
]

# Execute each script in order
for i, script in enumerate(scripts, start=1):
    print(f"\n🚀 Step {i}: Running {os.path.basename(script)}")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error in {os.path.basename(script)}:\n{result.stderr}")
        break
    else:
        print(f"✅ {os.path.basename(script)} completed successfully.\n")
