import streamlit as st
from pychord import Chord
import api
import os
import json

st.title('AI Music Generation Interface')

def load_model():
    selection = st.session_state['selected_model']

    old_selection = st.session_state.get('old_selection', None)
    
    if selection == old_selection:
        return
    else:
        print("Model changed from: ", old_selection, " to: ", selection)
        st.session_state['old_selection'] = selection

    selection = 'models/' + selection

    print("Loading model from: ", selection)

    try:
        with open(selection + '/details.json', 'r') as f:
            details = f.read()
            json_details = json.loads(details)

            print("Details: ", json_details)

        st.session_state['transformer'] = json_details["transformer"]
        st.session_state['timesteps'] = json_details["timesteps"]

        api.load_model_tokenizer(model_path=selection+'/'+json_details["model"],
                                tokenizer_path=selection+'/'+json_details["tokenizer"])
        
        print("Model loaded successfully")
    except:
        print("Error loading model: ", selection)
        st.error(f"Error loading model. Please check the details.json file in \"{selection}\" directory.")
        return

model_options = api.glob_models()

selection = st.selectbox('Select model', model_options, key='selected_model', on_change=load_model)

load_model()

# Scroll bar for BPM
bpm = st.slider('BPM', min_value=40, max_value=200, value=120)

# File uploader for MIDI files
uploaded_file = st.file_uploader("Upload your MIDI file", type=["mid", "midi"])

# Time input box
notes = st.number_input('number of notes: ', min_value=1, max_value=300, value=1)


if uploaded_file is None:
    # Toggle button for note or chord
    note_or_chord = st.radio('Select Note or Chord', ('Note', 'Chord'))
    
    # Dynamic dropdown based on toggle selection
    if note_or_chord == 'Note':
        options = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']
    else:
        options = ['C Major', 'D Minor', 'E Major', 'F Minor', 'G Major', 'A Minor', 'B Major']
        options_chord = ['C', 'Dm', 'E', 'Fm', 'G', 'Am', 'B']
    selection = st.selectbox('Select', options)

# Initialize session state variables
if "sheet_music" not in st.session_state:
    st.session_state['sheet_music'] = None
if "midi_file" not in st.session_state:
    st.session_state["midi_file"] = None
if "wav_file" not in st.session_state:
    st.session_state["wav_file"] = None
if "show_file_name_input" not in st.session_state:
    st.session_state["show_file_name_input"] = False  # Controls visibility of file name input
if "file_name" not in st.session_state:
    st.session_state["file_name"] = "generated_music"  # Default file name

# Create button
if st.button("Create"):
    if uploaded_file is not None:
        # Read the uploaded file
        midi_data = uploaded_file.read()

        with open("tmp.mid", "wb") as f:
            f.write(midi_data)
            
    def create_notes_from_selection():
        if uploaded_file is None:
            if note_or_chord == 'Note':
                note = selection
                return [note]
            else:
                index = options.index(selection)
                return Chord(options_chord[index]).components_with_pitch(root_pitch=4)

    if uploaded_file is None:
        api.create_midi_notes_from_alpha_notes(create_notes_from_selection())

    score = api.generate_score("tmp.mid", notes, bpm, st.session_state['timesteps'], st.session_state['transformer'])

    if os.path.exists("tmp.mid"):
            os.remove("tmp.mid")
    
    # Generate new MIDI and WAV files
    midi_file, wav_file, sheet_music = api.create_files(score)

    # Save the files in session state
    st.session_state["midi_file"] = midi_file
    st.session_state["wav_file"] = wav_file
    st.session_state['sheet_music'] = sheet_music

    # Reset file name input visibility when new files are created
    st.session_state["show_file_name_input"] = False


# Play the WAV file if available
if st.session_state["wav_file"]:
    st.audio(st.session_state["wav_file"], format="audio/wav")

if st.session_state['sheet_music']:
    if st.button('view sheet music'):
        try:
            st.session_state['sheet_music'].show()
        except:
            st.warning('cannot showcase music. consider downloading MuseScore, then run music21.configure.run().')

# Button to prompt MIDI file download
if st.session_state["midi_file"] and st.button("Download MIDI File?"):
    # Show the file name input field when download button is clicked
    st.session_state["show_file_name_input"] = True

# Show the file name input field if prompted
if st.session_state["show_file_name_input"]:
    file_name = st.text_input("Enter MIDI File Name (without extension)")

    # Display the actual download button when a valid file name is entered
    if file_name.strip():
        st.download_button(
            label="Download MIDI",
            data=st.session_state["midi_file"],
            file_name=f"{file_name}.mid",
            mime="audio/midi"
        )
