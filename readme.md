# Fake News Detection 📰🔍

Welcome to the Fake News Detection project! 🚀 This is a final-year project that uses machine learning to classify news as real ✅ or fake ❌ based on text input, uploaded videos, or YouTube URLs. Built with Streamlit for a slick UI, Whisper for audio transcription, and scikit-learn for powerful ML models, this app is your go-to tool for sniffing out fake news! 🕵️‍♂️

## Features

Text Input 📝: Type in news text and get instant classification.
Video Upload 🎥: Upload MP4 videos, extract audio, transcribe it, and classify the content.
YouTube URL 🌐: Paste a YouTube link to download, transcribe, and classify the video’s content.
Model Selection 🤖: Choose from four ML models:
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)

Explainability 🔎: See the top 5 words/phrases influencing the classification (for Logistic Regression and Random Forest).
User-Friendly UI 🎨: Built with Streamlit for a seamless experience.

## Prerequisites 🛠️

To run this project, you’ll need:

Python 3.8+ 🐍
A virtual environment (highly recommended) 🌍
FFmpeg installed for video processing (see FFmpeg Installation) 📼
The Full dataset.csv file for training models 📊
Pre-trained model pickle files (e.g., LogisticRegressionmodel.pkl) 📦

## Installation 📦

Clone the Repository:

```
git clone https://github.com/thisisfaizanali/fake-news-detection.git
cd fake-news-detection
```

## Set Up a Virtual Environment:

```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

## Install Dependencies:

```
pip install -r requirements.txt
```

## Add Dataset and Models:

Place Full dataset.csv in the project root (not included in Git due to size).
Add pre-trained pickle files (e.g., LogisticRegressionmodel.pkl, etc.) to the project root. These are not in the repo but can be regenerated using modelo.ipynb or obtained from the project owner.

## Run the App:

```
streamlit run app.py
```

## Usage 🚀

## Launch the App:

Run streamlit run app.py to open the Streamlit interface in your browser.

## Choose Input Type:

Select Text 📝, Video 🎥, or YouTube URL 🌐 from the sidebar.

## Pick a Model:

Choose a model (Logistic Regression, Decision Tree, Random Forest, or KNN) from the sidebar.

## Classify News:

Text: Enter news text and click "Classify".
Video: Upload an MP4 file to transcribe and classify.
YouTube URL: Paste a YouTube link to download, transcribe, and classify.

## View Results:

See if the news is Real ✅ or Fake ❌, with probability scores.
For Logistic Regression and Random Forest, check the top 5 influential words/phrases.

## Project Structure 📂

fake-news-detection/
├── app.py                # Main Streamlit app for classification 🌟
├── streamlitmain.py      # Handles video upload & transcription 🎥
├── yt_down_main.py       # Downloads & transcribes YouTube videos 🌐
├── modelo.ipynb          # Jupyter notebook for training ML models 📓
├── .gitignore            # Specifies files/folders to ignore in Git 🚫
├── README.md             # You're reading it! 📖
├── requirements.txt      # Python dependencies 📋
├── Full dataset.csv      # Dataset (excluded from Git repo) 📊
└── *.pkl                 # Pre-trained ML model files (excluded) 📦


## Dataset 📊

File: Full dataset.csv
Description: Contains news articles labeled as 0 (Fake) or 1 (Real).
Note: Excluded from the repository due to size. Contact the project owner or use a similar dataset. You can also regenerate it if you have access to the original sources.

## Model Training 🧠

Notebook: modelo.ipynb
Purpose: Loads and preprocesses the dataset, trains four ML models, and saves them as pickle files (commented out in the notebook).
Steps:
Run modelo.ipynb to preprocess Full dataset.csv.
Uncomment joblib.dump lines to save models as .pkl files.
Use these models in app.py for classification.

## Dependencies 📋

See requirements.txt for the full list. Key packages include:

streamlit 🎨: For the web interface
pandas 📊: For data handling
pytube 🌐: For YouTube video downloads
moviepy 🎥: For video/audio processing
whisper 🎙️: For audio transcription
scikit-learn 🤖: For ML models
numpy 🔢: For numerical operations
joblib 📦: For model persistence

Install them with:
pip install -r requirements.txt

## FFmpeg Installation 🖥️

FFmpeg is required for video/audio processing. Install it based on your OS:

Windows:
Download FFmpeg from ffmpeg.org.
Extract the archive and add the bin folder to your system PATH.
Verify with ffmpeg -version in your terminal.

MacOS:

```
brew install ffmpeg
```

Linux:

```
sudo apt-get install ffmpeg # Ubuntu/Debian
sudo yum install ffmpeg # CentOS/RHEL
```

## Notes ⚠️

Pickle Files: Pre-trained models (e.g., LogisticRegressionmodel.pkl) are not in the repo due to size. Regenerate them with modelo.ipynb or contact the project owner.
YouTube Downloads: Requires a stable internet connection and compliance with YouTube’s terms of service.
Whisper Model: Uses the base model. Ensure sufficient memory for video processing.
Virtual Environment: Strongly recommended to avoid dependency conflicts (see Installation).

## Contributing 🤝

Got ideas? 💡 Fork the repo, make changes, and submit a pull request! Follow PEP 8 guidelines and add clear comments. Let’s make this project even better! 🙌

## Acknowledgments 🙏

Built as a final-year project to combat fake news 📰.
Big thanks to the open-source communities behind Streamlit, Whisper, PyTube, and scikit-learn! ❤️
Special shoutout to my friend who helped with the ML bits! 🤗

Happy classifying! 🎉
