import streamlit as st
from transformers import pipeline
from transformers.pipelines import TextToAudioPipeline
import soundfile as sf
from scipy.signal import butter, lfilter
import io


# Define a low-pass filter for smoother and calmer audio
def low_pass_filter(data, cutoff, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


# Initialize text-to-audio pipeline
@st.cache_resource
def load_audio_pipeline():
    return pipeline("text-to-audio")  # Default: suno/bark-small


audio_pipeline: TextToAudioPipeline = load_audio_pipeline()

# Streamlit app
st.title("Text-to-Audio with Calm Effect")
text_input = st.text_area("Enter text to convert to audio:", "Hello world")

if st.button("Generate Audio"):
    with st.spinner("Generating audio..."):
        # Generate audio from text
        result = audio_pipeline(text_input)
        audio_data = result["audio"].flatten()
        sampling_rate = result["sampling_rate"]

        # Apply a low-pass filter for smoothness
        audio_data = low_pass_filter(audio_data, cutoff=4000, sampling_rate=sampling_rate)

        # Ensure 30 FPS (33.33 ms per frame)
        frame_duration = 1 / 30  # seconds
        frame_samples = int(sampling_rate * frame_duration)
        num_frames = len(audio_data) // frame_samples
        audio_data = audio_data[: num_frames * frame_samples]  # Trim excess samples

        # Save the processed audio to a file-like object
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sampling_rate, format="WAV")
        audio_buffer.seek(0)

        print("Sending audio to streamlit")
        # Display audio player
        st.audio(audio_buffer, format="audio/wav")
