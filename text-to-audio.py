from transformers import pipeline
from transformers.pipelines import TextToAudioPipeline
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

# Define a low-pass filter for smoother and calmer audio
def low_pass_filter(data, cutoff, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

# Initialize text-to-audio pipeline
audio_pipeline: TextToAudioPipeline = pipeline("text-to-audio")  # Default: suno/bark-small

# Generate audio from text
result = audio_pipeline("Hello world. This is a calm and collected message, spoken with clarity.")
audio_data = result["audio"].flatten()
sampling_rate = result["sampling_rate"]

# Apply a low-pass filter for smoothness
audio_data = low_pass_filter(audio_data, cutoff=4000, sampling_rate=sampling_rate)

# Adjust playback speed to 90% (slower and calmer)
speed_factor = 0.9
new_length = int(len(audio_data) / speed_factor)
audio_data = np.interp(
    np.linspace(0, len(audio_data), new_length),
    np.arange(len(audio_data)),
    audio_data
)

# Ensure 30 FPS (33.33 ms per frame)
frame_duration = 1 / 30  # seconds
frame_samples = int(sampling_rate * frame_duration)
num_frames = len(audio_data) // frame_samples
audio_data = audio_data[:num_frames * frame_samples]  # Trim excess samples

# Save the processed audio to a file
print("Saving audio to 'output_calmed.wav'")
sf.write("output_calmed.wav", audio_data, sampling_rate)
