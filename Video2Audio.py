from scipy import spatial  # for calculating vector similarities for search
from moviepy.editor import *
import spacy
from pyparsing import List
import tiktoken  # for counting tokens
import pandas as pd  # for storing text and embeddings data
import openai  # for calling the OpenAI API
import ast  # for converting embeddings saved as strings back to arrays
import operator
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd
from nudenet import NudeClassifier
from os import listdir
from ultralytics import YOLO
import os
import argparse
import time
import moviepy.editor as mp

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str) 

args = parser.parse_args()

os.mkdir(args.filename)

input = args.filename + '.mp4'
output = args.filename + '.mp3'

# Insert Local Video File Path
clip = mp.VideoFileClip(input) 
# Insert Local Audio File Path
clip.audio.write_audiofile(os.path.join(args.filename, output))
