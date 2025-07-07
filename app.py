import streamlit as st
import pandas as pd
import re
from pytube import YouTube
from moviepy.editor import VideoFileClip
import whisper
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

vectorizer=TfidfVectorizer()


def wordopt(text):
    text=text.lower()
    text=re.sub(r'https?://\S+|www\.\S+','',text)
    text=re.sub(r'<.*?>','',text)
    text=re.sub(r'[^\w\s]','',text)
    text=re.sub(r'\d','',text)
    text=re.sub(r'\n',' ',text)
    return text


news=pd.read_csv('Full dataset.csv')
news.reset_index(inplace=True)
news.drop(['index'],axis=1,inplace=True)
y=news['label']
news['text']=news['text'].apply(wordopt)
x=news['text']

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3)
xv_train=vectorizer.fit_transform(x_train)
xv_test=vectorizer.transform(x_test)

LR= LogisticRegression()
LR.fit(xv_train,y_train)

DT= DecisionTreeClassifier()
DT.fit(xv_train,y_train)
                    
RF= RandomForestClassifier()
RF.fit(xv_train,y_train)                    

KNN= KNeighborsClassifier()
KNN.fit(xv_train,y_train)


    
def classify_news(news, model):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorizer.transform(new_x_test)
    prediction = model.predict(new_xv_test)
    prob=model.predict_proba(new_xv_test)
    feature_name=vectorizer.get_feature_names_out()
    if hasattr(model,'coef_'):
        coeff=model.coef_[0]
    elif hasattr(model,'feature_importances_'):
        coeff=model.feature_importances_
    else:
        coeff=None
        
    return prediction[0], prob, coeff, feature_name


def extract_transcript(video_file):
    import tempfile
    import os
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())  # Write the uploaded file's contents
        temp_file_path = temp_file.name  # Get the temporary file path

    try:
        # Load the video from the temporary file path
        cvt_video = VideoFileClip(temp_file_path)
        ext_audio = cvt_video.audio
        ext_audio.write_audiofile("audio.mp3")
        cvt_video.close()  # Close the video file to free resources

        # Transcribe the audio using Whisper
        model = whisper.load_model("base")
        result = model.transcribe("audio.mp3")
        return result["text"]
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists("audio.mp3"):
            os.remove("audio.mp3")



def main():
    st.title("Fake and Real News Classifier")
    
    st.sidebar.title("Choose Input Type")
    input_type = st.sidebar.radio("Select input type:", ("Text", "Video","Youtube URL"))
    
    st.sidebar.title("Choose Model")
    model_option = st.sidebar.selectbox("Select Model:", ("Logistic Regression", "Decision Tree", "Random Forest", "K Nearest Neighbor"))
    
    if input_type == "Text":
        st.subheader("Text Input")
        text_input = st.text_area("Enter text:")
        
        if st.button("Classify"):
            if text_input.strip() != "":
                
                if model_option == "Logistic Regression":
                    prediction, probabilities, coefficients, feature_names = classify_news(text_input, LR)
                elif model_option == "Decision Tree":
                    prediction, probabilities, coefficients, feature_names = classify_news(text_input, DT)
                elif model_option == "Random Forest":
                    prediction, probabilities, coefficients, feature_names = classify_news(text_input, RF)
                elif model_option == "K Nearest Neighbor":
                    prediction, probabilities, coefficients, feature_names = classify_news(text_input, KNN)
                
                
                if prediction == 0:
                    st.write("This news is predicted to be **FAKE**.")
                    st.write("Probability of being FAKE:", probabilities[0][0])
                    st.write("Probability of being REAL:", probabilities[0][1])
                    if coefficients is not None:
                        top_feature_indices = coefficients.argsort()[-5:][::-1]
                        top_features = [feature_names[idx] for idx in top_feature_indices]
                        explanation = "The classification was influenced by the presence of the following significant words/phrases: "
                        explanation += ", ".join(top_features)
                        st.write(explanation)
                elif prediction ==1:
                    st.write("This news is predicted to be **REAL**.")
                    st.write("Probability of being FAKE:", probabilities[0][0])
                    st.write("Probability of being REAL:", probabilities[0][1])
                    
            else:
                st.warning("Please enter some text to classify.")



    elif input_type == "Video":
        st.subheader("Video Input")
        video_file = st.file_uploader("Upload a video file", type=["mp4"])
        
        if video_file is not None:
            video_transcript = extract_transcript(video_file)
            
            if model_option == "Logistic Regression":
                prediction, probabilities, coefficients, feature_names = classify_news(video_transcript, LR)
            elif model_option == "Decision Tree":
                prediction, probabilities, coefficients, feature_names = classify_news(video_transcript, DT)
            elif model_option == "Random Forest":
                prediction, probabilities, coefficients, feature_names = classify_news(video_transcript, RF)
            elif model_option == "K Nearest Neighbor":
                prediction, probabilities, coefficients, feature_names = classify_news(video_transcript, KNN)
                
                        
            if prediction == 0:
                st.write("This news is predicted to be **FAKE**.")
                st.write("Probability of being FAKE:", probabilities[0][0])
                st.write("Probability of being REAL:", probabilities[0][1])
                if coefficients is not None:
                    top_feature_indices = coefficients.argsort()[-5:][::-1]
                    top_features = [feature_names[idx] for idx in top_feature_indices]
                    explanation = "The classification was influenced by the presence of the following significant words/phrases: "
                    explanation += ", ".join(top_features)
                    st.write(explanation)
            elif prediction == 1:
                st.write("This news is predicted to be **REAL**.")
                st.write("Probability of being FAKE:", probabilities[0][0])
                st.write("Probability of being REAL:", probabilities[0][1])
                    
           
                
    elif input_type == "Youtube URL":
        yt_url = st.text_input("Upload URL")

        if st.button("Transcribe Video"):
            if yt_url:
                yt = YouTube(yt_url)
                video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                if video:
                    print(f"Downloading '{yt.title}'...")
                    video_path=video.download()
                    print("Download complete")
                else:
                    st.write("**Video not found**")
                
                
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
                    
                    if model_option == "Logistic Regression":
                        prediction, probabilities, coefficients, feature_names = classify_news(result, LR)
                    elif model_option == "Decision Tree":
                        prediction, probabilities, coefficients, feature_names = classify_news(result, DT)
                    elif model_option == "Random Forest":
                        prediction, probabilities, coefficients, feature_names = classify_news(result, RF)
                    elif model_option == "K Nearest Neighbor":
                        prediction, probabilities, coefficients, feature_names = classify_news(result, KNN)
                
                        
                    if prediction == 0:
                        st.write("This news is predicted to be **FAKE**.")
                        st.write("Probability of being FAKE:", probabilities[0][0])
                        st.write("Probability of being REAL:", probabilities[0][1])
                        if coefficients is not None:
                            top_feature_indices = coefficients.argsort()[-5:][::-1]
                            top_features = [feature_names[idx] for idx in top_feature_indices]
                            explanation = "The classification was influenced by the presence of the following significant words/phrases: "
                            explanation += ", ".join(top_features)
                            st.write(explanation)
                    elif prediction == 1:
                        st.write("This news is predicted to be **REAL**.")
                        st.write("Probability of being FAKE:", probabilities[0][0])
                        st.write("Probability of being REAL:", probabilities[0][1])
                    



if __name__ == "__main__":
    main()