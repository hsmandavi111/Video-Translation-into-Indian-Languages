! pip install git+https://github.com/openai/whisper.git -q requests gradio -q  pysrt  webvtt-py vtt_to_srt3  gTTS pydub

from gtts import gTTS
import IPython.display as ipd
import gradio as gr 
import os
import sys
import subprocess
from whisper.utils import get_writer
import whisper
import json
import requests
import webvtt
from transformers import pipeline
import cv2
import pysrt
import datetime
import subprocess
import ffmpeg
from pydub import AudioSegment
import re

# clone the repo for running evaluation
!git clone https://github.com/AI4Bharat/indicTrans.git
%cd indicTrans
# clone requirements repositories
!git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
!git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
!git clone https://github.com/rsennrich/subword-nmt.git
%cd ..


# Install the necessary libraries
!pip install sacremoses pandas mock sacrebleu tensorboardX pyarrow indic-nlp-library
! pip install mosestokenizer subword-nmt
# Install fairseq from source
!git clone https://github.com/pytorch/fairseq.git
%cd fairseq
# !git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
!pip install ./
! pip install xformers
%cd ..


# add fairseq folder to python path
os.environ['PYTHONPATH'] += ":/content/fairseq/"
# sanity check to see if fairseq is installed
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils


# download the indictrans model

# downloading the indic-en model
# !wget https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/indic2en.zip
# !unzip indic2en.zip

# downloading the en-indic model
!wget https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/en2indic.zip
!unzip en2indic.zip

# # downloading the indic-indic model
# !wget https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/m2m.zip
# !unzip m2m.zip

%cd indicTrans


from indicTrans.inference.engine import Model

indic2en_model = Model(expdir='../en-indic')

model = whisper.load_model("medium")


# function to Extract audio from video 
def video2mp3(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return f"{filename}.{output_ext}"
    
 
 # Function to translate audio and create subtittle file from the audio where input file is video file

def subtittle(input_video):

    audio_file = video2mp3(input_video)
    print(audio_file)
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(audio_file,**translate_options)

    #  ouput directory to save subtittle file   
    output_dir = '/content/'
    audio_path = audio_file.split(".")[0]

    # writer = get_writer("vtt", output_dir)
    # writer(result, audio_path)

    return result
    
    
    
    
    
    
    
def trans(result,lang1,lang2):
  en_text_arr = [0]
  for i in range(len(result['segments'])):
    en_text_arr[0] = result['segments'][i]['text']
    out = indic2en_model.batch_translate(en_text_arr, 'en', 'ta')
    result['segments'][i]['text'] = out[0]
  
  return result




def SRT(result,dir):
  writer = get_writer("srt", output_dir)
  writer(result, 'fname')
  srtfile = dir +  'fname' + '.srt'
  return srtfile
  
  
 
def text_2_speech(srt_path,lang_code,output_dir):
    # Read the subtitle file
    with open(srt_path, "r") as file:
        subtitles = file.read()

    # Use regular expressions to extract the text and timestamps
    subtitle_pattern = re.compile(r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|$)")
    subtitles = subtitle_pattern.findall(subtitles)

    # Create an empty audio file
    audio = AudioSegment.empty()

    # Convert the subtitles to audio and add them to the audio file
    for subtitle in subtitles:
        # Get the start and end times for this subtitle
        start_time = subtitle[1]
        end_time = subtitle[2]

        x = subtitle[1].split(",")[0]
        y = int(x.split(":")[0])*60*60 + int(x.split(":")[1])*60 + int(x.split(":")[2])
        y = y*1000

        x = subtitle[2].split(",")[0]
        y2 = int(x.split(":")[0])*60*60 + int(x.split(":")[1])*60 + int(x.split(":")[2])
        y2 = y2*1000

        d = y2-y 
        # Get the text for this subtitle
        text = subtitle[3]

        # Generate the audio for this subtitle
        tts = gTTS(text, lang= lang_code)
        dir = output_dir + 'temp.mp3'
        tts.save(dir)
        segment = AudioSegment.from_file(dir, format="mp3")
        
        # compress the segment to maintain the lenthg of speaked text
        cmprs_segment = segment[:d]
        # Add the audio to the final audio file
        audio = audio + cmprs_segment

    # Save the final audio file
    dir2 = output_dir + 'cmprsd' + 'temp.mp3'
    audio.export(dir2, format="mp3")
    return dir2
    
    
    

def concat_aud_vid(aud_path,vid_path,dir):
    
    input_video = ffmpeg.input(vid_path)
    added_audio = ffmpeg.input(aud_path).audio.filter('adelay', "1500|1500")

    vid_dir = dir + 'tran_vid.mp4'
    
    # merged_audio = ffmpeg.filter([input_video.audio, added_audio], 'amix')

    (
        ffmpeg
        .concat(input_video, added_audio, v=1, a=1)
        .output(vid_dir)
        .run(overwrite_output=True)
    )
    return vid_dir
    
    
# generate  text file from video in english
result = subtittle('/content/Diamond back moth.mp4')

# source language code-2
lang1 = 'en' 
# target language code-2
lang2 = 'ta' 

# translate text file into target languages
trans_result = trans(result,lang1,lang2)


# set output directive
output_dir = '/content/'

srt_path = SRT(trans_result,output_dir)

lang_code = 'ta'
audio_path = text_2_speech(srt_path,lang_code,output_dir)

vid_path = '/content/Diamond back moth.mp4'
trans_path =  concat_aud_vid(audio_path,vid_path,output_dir)

print(trans_path)
