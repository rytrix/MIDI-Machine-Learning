# MIDI-Machine-Learning
Using machine learning to generate MIDI piano music

# Description
This code has all the tools needed to create an AI application. The python notebooks consist of all the code needed to create a music generation model,
which can then be easily integrated within the application. 

# Requirements

### Linux Requirements
```bash
sudo apt install fluidsynth
```
### Python virtual environment
```bash
python -m venv *directory*
source *directory*/bin/activate
```
### Base Requirements
```bash
pip install -r requirements.txt
```
### GPU Dependencies
```bash
pip install -r requirements-tensorflow-cuda.txt
```

# Usage

### Training
Models can be trained through the "lstm.ipynb" and "transformer.ipynb" python notebooks. First however, midi training data should be downloaded, a small script in "download_midis.ipynb" is available to help. After these models are trained, the GUI expects a specific file format to parse. The last cell in these python notebooks has a function to generate a directory structure expected by the GUI. These can be moved into a directory named "models" to show up in the GUI.

### GUI
The GUI can be run with ```streamlit run gui.py``` 
within your python environment and provided the "models/" directory is present it will pick up on available models that have been moved into that directory.

# Midi Training Data Source Source
https://github.com/bytedance/GiantMIDI-Piano

# Contributors
This project was a part of a group capstone, meaning there are and will only be the current three contributiors for this project. 
