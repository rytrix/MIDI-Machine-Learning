import pretty_midi
import numpy as np
import keras
from miditok import REMI
import io
import os
from midi2audio import FluidSynth
from symusic.core import TempoTick
from music21 import converter, midi
import tensorflow as tf

soundfont_path = "AMS_Steinway_Grand.sf2"

def glob_models(models_dir = "models"):
    """
    List all directories in the specified models directory.

    Parameters:
    - models_dir (str): The path to the models directory (default is "models").

    Returns:
    - list: A list of directory names in the models directory.
    """
    if not os.path.exists(models_dir):
        print(f"Directory '{models_dir}' does not exist.")
        return []

    # List all directories in the models directory
    directories = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    return directories

def load_model_tokenizer(model_path = "model.keras", tokenizer_path = "tokenizer.json"):
    """
    Load the model and tokenizer from the specified paths.
    """
    print("Loading model from: ", model_path)
    global model
    model = keras.models.load_model(model_path)
    print("Loading tokenizer from: ", tokenizer_path)
    global tokenizer
    tokenizer = REMI(params=tokenizer_path)

def create_midi_notes_from_alpha_notes(notes, output = "tmp.mid"):
    """
    Create a MIDI file from a list of note names.

    Parameters:
    - notes (list): A list of note names (e.g., ['C4', 'E4', 'G4']).
    - output (str): The output file path for the generated MIDI file (default is "tmp.mid").

    This function converts a list of note names into MIDI notes, assigns them to a piano instrument,
    and writes the resulting MIDI data to the specified output file.
    """
    piano_object = pretty_midi.PrettyMIDI()

    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano') #or "Piano"
    piano = pretty_midi.Instrument(program=piano_program)

    # Iterate over note names, which will be converted to note number later
    i = 0
    for note_name in notes:
        # Retrieve the MIDI note number for this note name
        note_number = pretty_midi.note_name_to_number(note_name)
        # Create a Note instance, starting at 0s and ending at .5s
        note = pretty_midi.Note(
            velocity=100, pitch=note_number, start=i, end=i + 0.5)
        # Add it to our piano instrument
        piano.notes.append(note)
        i += 0.5

    # Add the piano instrument to the PrettyMIDI object
    piano_object.instruments.append(piano)
    piano_object.write(output)

def generate_score(given_notes, num_notes_to_generate, bpm, timesteps=50, is_transformer=False):
    """
    Generate a sequence of MIDI notes based on given input notes using a trained model.

    Parameters:
    - given_notes (list): A list of initial notes to seed the generation process.
    - num_notes_to_generate (int): The number of notes to generate.
    - bpm (int): The tempo in beats per minute for the generated score.
    - timesteps (int): The timesteps the model was trained on

    Returns:
    - score: The generated score with the specified tempo.
    """
    generated_notes = []
    x = tokenizer.encode(given_notes)
    seed = np.array(x[0:timesteps])

    if is_transformer:
        seed = tf.pad(seed, paddings=[[timesteps-len(seed), 0]], constant_values=0)
        # print(seed)
        for _ in range(num_notes_to_generate):
            seed = np.reshape(seed, (1, len(seed)))
            seed = tf.convert_to_tensor(seed, dtype=tf.int32)
            prediction = model.predict(seed, verbose=0)
            prediction = np.argmax(prediction, axis=1)
            # print(prediction)

            seed = np.append(seed.numpy(), prediction)[-timesteps:]
            
            generated_notes.append(int(prediction[0]))
        # print(generated_notes)
    else:
        for _ in range(num_notes_to_generate):
            seed = np.reshape(seed, (1, len(seed), 1))
            prediction = np.argmax(model.predict(seed, verbose=0))
            
            seed = np.append(seed, prediction)[len(seed)-timesteps:]
            
            generated_notes.append(int(prediction))
    
    score = tokenizer(generated_notes)

    new_tempo = TempoTick(time=0, qpm=bpm)  # Time is in ticks, qpm is the tempo

    score.tempos[0] = new_tempo
    # print(score)

    return score


def create_files(score):
    """
    Generate a MIDI and a WAV file

    Parameters:
    - score

    Returns:
    - midi_file
    - wav_file
    """
    
    midi_path = 'temp.mid'

    midi = score.dump_midi(midi_path)

    # midi file in memory
    midi_file = io.BytesIO()

    with open(midi_path, "rb") as temp_midi:
        midi_file.write(temp_midi.read())
    midi_file.seek(0)  # Reset pointer for reading

    midi_data = midi_file.getvalue()  # Raw MIDI bytes

    sheet_music = converter.parse(midi_path)

    wav_file = midi_to_wav(midi_data, temp_midi, midi_path)

    return midi_file, wav_file, sheet_music
    


def midi_to_wav(midi_data, temp_midi, temp_midi_path):
    """
    Convert MIDI data to a WAV file with temporary disk storage.

    Parameters:
     - midi_data (bytes): The raw MIDI data as bytes.
     - temp_midi: the actual midi file as a variable.
     - temp_midi_path: pathway to a temporary midi file on disk. (This is to ensure initialization)

    Returns:
     - wav_file: The WAV file stored in memory.
    """
    # Define temporary file paths
    temp_wav_path = "temp.wav"

    try:
        # Initialize FluidSynth with the specified SoundFont
        fs = FluidSynth(soundfont_path)

        # Convert the temporary MIDI file to a temporary WAV file
        fs.midi_to_audio(temp_midi_path, temp_wav_path)

        # Read the WAV file into memory
        wav_file = io.BytesIO()
        with open(temp_wav_path, "rb") as temp_wav:
            wav_file.write(temp_wav.read())
        wav_file.seek(0)  # Reset pointer for reading

    finally:
        # Ensure cleanup of temporary files
        if os.path.exists(temp_midi_path):
            os.remove(temp_midi_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    # Return the WAV file stored in memory
    return wav_file

if __name__ == "__main__":
    # models = glob_models()
    # print(models)

    pass