import streamlit as st
from pytube import YouTube
from moviepy.editor import VideoFileClip
import whisper
import os


def main():
    st.title("Video Transcription")

    yt_url = st.text_input("Upload URL")

    if st.button("Transcribe Video"):
        if yt_url:
            st.write("Video downloading")
            
            yt = YouTube(yt_url)
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if video:
                print(f"Downloading '{yt.title}'...")
                video_path=video.download()
                    
                
                print("Download complete.")
            else:
                print("Video not found.")
            
            
            if video_path:
                st.write("Transcribing, Please wait.....")
                
                cvt_video = VideoFileClip(video_path)
                ext_audio = cvt_video.audio
                ext_audio.write_audiofile("audio.mp3")

                model = whisper.load_model("base")
                result = model.transcribe("audio.mp3")
                result=result["text"]

                os.remove("audio.mp3")
                
                st.text_area("Transcription",result,height=200)
           
                cvt_video.close()
                os.remove(video_path)
                
    
if __name__ == "__main__":
    main()
