import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
import soundfile as sf

def load_audio_files(directory, sr=22050, duration=2.0):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
    audio_data = []
    for f in files:
        audio, _ = librosa.load(f, sr=sr, duration=duration)
        if len(audio) < sr * duration:
            audio = np.pad(audio, (0, int(sr * duration) - len(audio)), 'constant')
        else:
            audio = audio[:int(sr * duration)]
        audio_data.append(audio)
    return np.array(audio_data)

def augment_audio(audio, sr, duration=2.0):
    augmented = []
    target_length = int(sr * duration)
    for a in audio:
        # Original
        augmented.append(a)
        
        # Pitch shifting
        pitch_shifted = librosa.effects.pitch_shift(a, sr=sr, n_steps=np.random.uniform(-2, 2))
        augmented.append(pad_or_truncate(pitch_shifted, target_length))
        
        # Time stretching
        time_stretched = librosa.effects.time_stretch(a, rate=np.random.uniform(0.8, 1.2))
        augmented.append(pad_or_truncate(time_stretched, target_length))
        
        # Adding noise
        noise = np.random.randn(len(a))
        augmented.append(pad_or_truncate(a + 0.005 * noise, target_length))
        
    return np.array(augmented)

def pad_or_truncate(audio, target_length):
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), 'constant')
    else:
        return audio[:target_length]

def extract_stft(audio_data, n_fft=2048, hop_length=512):
    stft_magnitude = []
    phase = []
    for audio in audio_data:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase_comp = librosa.magphase(stft)
        stft_magnitude.append(magnitude)
        phase.append(phase_comp)
    return np.array(stft_magnitude), np.array(phase)

# Step 1: Prepare the Dataset
audio_data = load_audio_files('01-wav')

# Augment the Dataset
augmented_audio_data = augment_audio(audio_data, sr=22050)

# Step 2: Extract STFT Magnitude
stft_magnitude, phase = extract_stft(augmented_audio_data)
input_shape = stft_magnitude.shape[1:]  # Adjust shape based on your STFT magnitude feature size

# Step 3: Train a Variational Autoencoder (VAE)
# Parameters
latent_dim = 16  # Dimensionality of the latent space

# Encoder
inputs = Input(shape=input_shape)
x = LSTM(128, activation='relu', return_sequences=False)(inputs)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_input = Input(shape=(latent_dim,))
x = Dense(128, activation='relu')(decoder_input)
x = RepeatVector(input_shape[1])(x)
x = LSTM(128, activation='relu', return_sequences=True)(x)
x = TimeDistributed(Dense(input_shape[0], activation='relu'))(x)
outputs = Reshape(input_shape)(x)

# Models
encoder = Model(inputs, [z_mean, z_log_var, z])
decoder = Model(decoder_input, outputs)
vae_outputs = decoder(encoder(inputs)[2])

# VAE model
vae = Model(inputs, vae_outputs)

# Custom loss layer
class VAE_LossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VAE_LossLayer, self).__init__(**kwargs)

    def vae_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = MeanSquaredError()(inputs, outputs)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    def call(self, inputs):
        inputs, outputs, z_mean, z_log_var = inputs
        loss = self.vae_loss(inputs, outputs, z_mean, z_log_var)
        self.add_loss(loss)
        return outputs

loss_layer = VAE_LossLayer()([inputs, vae_outputs, z_mean, z_log_var])
vae = Model(inputs, loss_layer)

vae.compile(optimizer='adam')
vae.summary()

# Train the VAE
vae.fit(stft_magnitude, stft_magnitude, epochs=100, batch_size=32)

# Step 4: Generate New Fart Sounds
def generate_fart_sound(decoder, n_samples=1):
    z_sample = np.random.normal(size=(n_samples, latent_dim))
    generated_stft_mag = decoder.predict(z_sample)
    return generated_stft_mag

generated_stft_mag = generate_fart_sound(decoder)

# Ensure STFT magnitudes are finite
generated_stft_mag = np.nan_to_num(generated_stft_mag)

# Convert STFT magnitudes back to audio
def stft_to_audio(magnitude, phase, hop_length=512):
    stft = magnitude * phase
    audio = librosa.istft(stft, hop_length=hop_length)
    return audio

# Define the sample rate
sample_rate = 22050

# Use the first phase component (assuming all have the same phase)
for i, mag in enumerate(generated_stft_mag):
    audio = stft_to_audio(mag, phase[0], hop_length=512)
    sf.write(f'generated_fart_{i}.wav', audio, sample_rate)