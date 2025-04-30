from miditok import REMI, TokenizerConfig
from pathlib import Path
import numpy as np
import keras
import talos

# ### Load a tokenizer
tokenizer = REMI(params="tokenizer.json")
paths_midis = list(Path("midis_train").glob('**/*.mid'))
paths_midis.extend(list(Path("midis_test").glob('**/*.mid')))
paths_midis.extend(list(Path("midis_val").glob('**/*.mid')))

# ### Setup training data
from helpers import load_songs_random, setup_timestep_data

paths_train = list(Path("midis_train").glob('**/*.mid'))
paths_val = list(Path("midis_val").glob('**/*.mid'))

# Load in randomly selected songs
train_data = load_songs_random(tokenizer, paths_train, 5)
val_data = load_songs_random(tokenizer, paths_val, 2)

vocab_size = tokenizer.vocab_size
timesteps = 50

x_train, y_train = setup_timestep_data(train_data, timesteps)
x_val, y_val = setup_timestep_data(val_data, timesteps)

p = {
    'units': [128, 256, 512, 1024],
    'dropout': (0.0, 0.5, 0.1),
    'learning_rate': [0.001, 0.01, 0.05]
}

def piano_model(x_train, y_train, x_val, y_val, params):
    # ### Create a model
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, 256),
        keras.layers.LSTM(params['units'], recurrent_dropout=params['dropout'], return_sequences=True),
        keras.layers.Dropout(params['dropout']),
        keras.layers.LSTM(params['units'], recurrent_dropout=params['dropout']),
        keras.layers.Dropout(params['dropout']),
        keras.layers.Dense(vocab_size, activation='softmax'),
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=params['learning_rate']), metrics=['accuracy'])
    model.build(input_shape=(timesteps, 1))

    # Fit the model
    out = model.fit(x_train, y_train, epochs=60, validation_data=(x_val, y_val), callbacks=keras.callbacks.EarlyStopping(min_delta=0.001, patience=3, restore_best_weights=True))
    return out, model

talos.Scan(x_train, y_train, model=piano_model, params=p, experiment_name='experiment1', reduction_method='correlation', reduction_interval=2, reduction_window=2, reduction_metric='val_loss')