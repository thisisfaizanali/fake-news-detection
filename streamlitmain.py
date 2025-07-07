import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
import os

def transcribe_video(video_file):
    # Load the video and extract the audio
    cvt_video = VideoFileClip(video_file)
    ext_audio = cvt_video.audio
    ext_audio.write_audiofile("audio.mp3")

    # Transcribe the audio using the Whisper model
    model = whisper.load_model("base")
    result = model.transcribe("audio.mp3")

    # Remove the temporary audio file
    os.remove("audio.mp3")

    return result["text"]

def main():
    st.title("Video Transcription")

    # Upload video file
    uploaded_file = st.file_uploader("Upload video file", type=["mp4"])

    if uploaded_file is not None:
        st.write("Video uploaded successfully!")
        st.write("Converting, Please wait.....")

        # Perform transcription when video is uploaded
        result = transcribe_video(uploaded_file.name)

        # Display the transcription result
        st.header("Transcription Result")
        st.text_area("Transcription", result, height=200)

        # Allow user to download the transcript text
        st.download_button(
            label="Download Transcript",
            data=result,
            file_name="transcript.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
