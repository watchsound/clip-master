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

from VideoComponents import SRT
from VideoComponents import VideoScene
from VideoComponents import MergedScene
from VideoComponents import TimeRangeUtils
from VideoComponents import ContextScene

import ChatGPTUtils
import IOUtils
import VideoComponents
from StringUtils import StringUtils

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
#output video time in second
parser.add_argument('--length', type=int, required=True)

parser.add_argument('--climax', type=int)

parser.add_argument('--sentiment', type=float)

parser.add_argument('--sex', type=float)
 
args = parser.parse_args()
 
#load scenes file which is returned from gpt
with open(args.filename + '-gptscenes.txt', 'r') as file:
    data = file.read()
    scenes = ChatGPTUtils.getContextSceneBy(data)
    print(f" we have {len(scenes)} scenes")

#load srt file
srt_orig = IOUtils.load_srt(args.filename + '.srt')
print(f" we have {len(srt_orig)} srts")

#decompose video
#we dont use  split-video here for now
####os.system('scenedetect -i ' + args.filename + '.mp4 -o ' +
####          args.filename + ' detect-adaptive list-scenes save-images')

print(f" we have done scene detect  ")

#parse v-scene data
video_scenes = IOUtils.load_video_scenes(
    './' + args.filename + '/' + args.filename + '-Scenes.csv')
print(f" we have {len(video_scenes)} video scenes from scenedetect")

video_total_length = video_scenes[len(video_scenes)-1].end

#add object to v-scene images
objRecModel = YOLO("yolov8m.pt")
image_files = [f for f in listdir(args.filename) if '.jpg' in f]
count = 0
for f in image_files:
    ####
    ###continue
    results = objRecModel.predict('./' + args.filename + '/' + f)
    objs = ''
    persons  = 0
    for re in results:
        for bx in re.boxes:
            obj = re.names[bx.cls[0].item()]
            if obj == 'person':
                persons += 1
            else:
                objs += ' ' + obj
    video_scenes[count//3].objects = objs
    video_scenes[count//3].persons = persons
    count += 1
print(f" we have added objects to video scenes")


#add nude detecting..
classifier = NudeClassifier()
count = 0
for f in image_files:
    ####
    ###continue
    vscene = video_scenes[count//3]
    if vscene.persons == 1 or vscene.persons == 2 :
        fn = './' + args.filename + '/' + f
        results = classifier.classify(fn)
        vscene.sexValue = results[fn]['unsafe']
    count += 1
print(f" we have added nudelity level to video scenes")


#from v-scenes get cluster dialogs
#use srt to adjust v-scenes (merge v-scenes if belong to one sentence in srt)
merged_scenes = IOUtils.merge_scenes(srt_orig, video_scenes, 25)
print(f" we have {len(merged_scenes)} video scenes after merge")
 

#for ms in merged_scenes:
#   print(f" order {str(ms.order)}  start {str(ms.start)}  end {str(ms.end)} ")
#   print(f" {ms.mergedContent }")

# add sentiment score
sia = SentimentIntensityAnalyzer()
for ms in merged_scenes:
    ms.sentiments = sia.polarity_scores(ms.mergedContent)["compound"]
print(f" we have finished sentiment annotations")


#ask chatgpt to find best match cluster for each scene
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


scene_match_prompt = "i will give you several hundrend dialogs from a play, please find top three dialogs which best match the scene without explanation:"
scene_match_prompt2 = "\nplease response with format:\nbest match with original line in dialogs including leading row number\nsecond best match with original line in dialogs including leading row number\nthird best match with original line in dialogs including leading row number"
token_budget = 4096 - 1000

context_scenes_count = len(scenes)
merged_scenes_count = len(merged_scenes)


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_top_three_pos(text: str, vscenes: List[MergedScene], scene: ContextScene):
    """   top three index in scene """
   # print(f"find the top three pos")
    for aline in text.split("\n"):
        if len(aline) == 0:
            continue;
        row = StringUtils().checkNumberRow(aline)
       # print(f"row = {row}")
        if row < 0:
            if aline.startswith('Best match:') or aline.startswith("Second best match:") or aline.startswith("Third best match:"):
                poss = aline.find(":")
                aline = aline[poss+1: min(poss+30, len(aline))].strip().replace("\"", "")
                if len(aline) == 0:
                    continue
                #print(f"#{aline}#")
                for ascene in vscenes:
                    if ascene.mergedContent is not None and ascene.mergedContent.find(aline) >= 0:
                        scene.matchedDialogOrders.append(ascene.order)
                        break
            continue
        scene.matchedDialogOrders.append(row)
        # if aline.startswith("1.") or aline.startswith("2.") or aline.startswith("3."):
        #    aline = aline[3:]
        #    for ascene in vscenes:
        #       if ascene.mergedContent.index(aline) >= 0:
        #          scene.matchedDialogOrders.append(ascene.order)
        #          break




def get_scene_match_by_gpt0(ascene: ContextScene, fstart: int, fend: int):
    fprompt = scene_match_prompt + " " + \
        ascene.summary + " " + ascene.description + "\n\n" + scene_match_prompt2  + "\n\n"

    for vpos in range(fstart, fend):
        mscene = merged_scenes[vpos]
        if mscene.mergedContent is None or len(mscene.mergedContent) == 0:
            continue;
        ccc = '\n' + str(mscene.order) + ". " + mscene.mergedContent
        if (
            num_tokens(fprompt + ccc, model=GPT_MODEL)
            > token_budget
        ):
            break
        else:
            fprompt += ccc

  #  print(fprompt)
    messages = [
        {"role": "system", "content": "You are a movie or play writer or director."},
        {"role": "user", "content": fprompt},
    ]
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
 #   print(response_message)
    get_top_three_pos(response_message, merged_scenes, ascene)
    print(f"best match scene index is {ascene.matchedDialogOrders}")
  


decomposed_chunk_size = 4
if context_scenes_count < 9:
    decomposed_chunk_size = 2
decomposed_chunk = merged_scenes_count // decomposed_chunk_size


def get_scene_match_by_gpt(ascene: ContextScene, fstart: int, fend: int):
    get_scene_match_by_gpt0(ascene, fstart, fend)
         

    if len(ascene.matchedDialogOrders) < 3:
        get_scene_match_by_gpt0(
            ascene, int(max(0, fstart - decomposed_chunk/2)), int(min(fend +
                                                             decomposed_chunk/2, len(merged_scenes))))

    if len(ascene.matchedDialogOrders) < 3:
        ascene.matchedDialogOrders.append(0)
    if len(ascene.matchedDialogOrders) < 3:
        ascene.matchedDialogOrders.append(0)
    if len(ascene.matchedDialogOrders) < 3:
        ascene.matchedDialogOrders.append(0)
#for index, ascene in enumerate(scenes):



fstart = 0
fend = decomposed_chunk

def midvalue(values):
    minv = min(values)
    maxv = max(values)
    for v in values:
        if v!= minv and v != maxv:
            return v
    return values[0]
    

 
# from start to middle
for pos in range(0,  context_scenes_count // 2 + 1):
    ascene = scenes[pos]
    if pos > 0:
        fstart = midvalue(scenes[pos-1].matchedDialogOrders) 
    fend = min(fstart + decomposed_chunk, merged_scenes_count)
    get_scene_match_by_gpt(ascene, fstart, fend)  
    time.sleep(30)

#from end to middle
fstart = merged_scenes_count  - decomposed_chunk
fend = merged_scenes_count  
for pos in range(context_scenes_count - 1,  context_scenes_count // 2 - 1, -1):
    ascene = scenes[pos]
    if len(ascene.matchedDialogOrders) > 0:
        continue
    if pos < context_scenes_count - 1:
      #  fend = min(fend, max(scenes[pos+2].matchedDialogOrders))
        fend = midvalue(scenes[pos+1].matchedDialogOrders)
    fstart = max(fend - decomposed_chunk, 0)
    get_scene_match_by_gpt(ascene, fstart, fend)
    time.sleep(30)

#for the midle one
if context_scenes_count % 2 == 1:
    pos = context_scenes_count // 2 + 1
    ascene = scenes[pos]
    if len(ascene.matchedDialogOrders) == 0: 
        get_scene_match_by_gpt(ascene, merged_scenes_count //
                           2 - decomposed_chunk, merged_scenes_count//2 + merged_scenes_count)   
 
""" testdata = [
  [15, 17, 31], 
  [31, 32, 67] ,
  [32, 94, 96],
  [97, 109, 106],
  [106, 109, 116],
  [109, 137, 138, 116],
   [140, 109, 137],
   [168, 167, 174],
  [177, 198, 210, 231],
  [202, 240, 259],
]

for i in range( len(scenes) ):
    scenes[i].matchedDialogOrders = testdata[i]   """

#print out for debug purpose
for ascene in scenes:
    print(f" scene name :   {str(ascene.order)}  {ascene.summary} {ascene.matchedDialogOrders} ")


def max_word_sim_score(doc1, doc2) -> float:
    """ ."""
    maxvalue = 0
    for token1 in doc1:
        for token2 in doc2:
            v = token1.similarity(token2)
            if v > maxvalue:
                maxvalue = v
    return maxvalue


#adjust matches by object in image to context scene
nlp = spacy.load('en_core_web_md')
object_weight = 1

for ascene in scenes:
    if len(ascene.suggestObjects) == 0:
        continue
    scenedoc = nlp(ascene.suggestObjects)
    for p in range(3):
        mergedscene = merged_scenes[ascene.matchedDialogOrders[p]]
        if len(mergedscene.mergedObjects) > 0:
            mergedscenedoc = nlp(mergedscene.mergedObjects)
            ascene.matchedDialogScore[p] += max_word_sim_score(scenedoc,
                                                               mergedscenedoc) * object_weight

#adjst matches by sentiment
sentiment_weight = 1
if args.sentiment is not None:
    sentiment_weight = args.sentiment
sex_weight = 0.5
if args.sex is not None:
    sex_weight = args.sex

for ascene in scenes:
    for p in range(3):
        mergedscene = merged_scenes[ascene.matchedDialogOrders[p]]
        ascene.matchedDialogScore[p] += abs(
            mergedscene.sentiments) * sentiment_weight
        if mergedscene.mergedSex > 0.6:
            ascene.matchedDialogScore[p] += abs(
                mergedscene.mergedSex) * sex_weight


for ascene in scenes:
    print(
        f" order {str(ascene.order)}  {str(ascene.matchedDialogOrders[0])}  {str(ascene.matchedDialogScore[0])}  {str(ascene.matchedDialogOrders[1])}  {str(ascene.matchedDialogScore[1])}   {str(ascene.matchedDialogOrders[2])}  {str(ascene.matchedDialogScore[2])}")

#find best path
ContextScene.dyBestPath(scenes)
for ascene in scenes:
    print(f" path: {str(ascene.bestDialogOrder)}")

#merge new video
clip = VideoFileClip(args.filename + ".mp4")
clips = []
total_output_time = args.length
if total_output_time <= 5:
    total_output_time = 180  # 3 minutes
ascene_time = total_output_time / len(scenes)

print(f" ascene_time {str(ascene_time)}") 

#find most sex one
if sex_weight > 0.1:
    sex_mscene = None
    sex_mscene_value = 0 
    for mscene in merged_scenes:
        if (mscene.persons == 1 or mscene.persons == 2) and mscene.mergedSex > sex_mscene_value :
            sex_mscene = mscene
            sex_mscene_value = mscene.mergedSex
    

acc_time = 0
acc_times = []
prev_bestorder = -111 
for ascene in scenes:
    cborder = ascene.bestDialogOrder
    if prev_bestorder == cborder:
        ascene.repeatCount += 1
        continue 
    mergedscene = merged_scenes[cborder]
    print(
        f" mergedscene : {str(mergedscene.start)}   {str(mergedscene.end)}")
    acc_time += mergedscene.end - mergedscene.start 
    prev_bestorder = cborder

print(f" acc_time {str(acc_time)}")

adjusted_ascene_time_margin = (total_output_time - acc_time) / len(scenes)
print(f" adjusted_ascene_time_margin {str(adjusted_ascene_time_margin)}")


pos = 0
prestart = 0
while pos < len(scenes):
    ascene = scenes[pos] 
    if ascene.bestDialogOrder < 0:
        print(
            f" a scene can not match into story, scene order is {ascene.order} scene name is {ascene.summary}")
        pos += 1
        continue 
    mergedscene = merged_scenes[ascene.bestDialogOrder]

    if sex_mscene is not None and prestart > 0 and sex_mscene.start > prestart and sex_mscene.end < mergedscene.start:
        clips.append(clip.subclip(sex_mscene.start, sex_mscene.end))
        print(
            f" sexest one : mstart = {str(sex_mscene.start)}   mend = {str(sex_mscene.end)}")
        prestart = sex_mscene.end


    # TODO ... make life simple, we wont do tuning here
    #while acc_time + mergedscene.end - mergedscene.start <= ascene_time:
   # margin = (ascene_time - mergedscene.end + mergedscene.start)/2
    arange = mergedscene.end - mergedscene.start
    if adjusted_ascene_time_margin < 0 and (arange + adjusted_ascene_time_margin)/arange < 0.8:
        margin = 0
    else:
        margin =  adjusted_ascene_time_margin /2
       # margin = max(0, margin)
    mstart = max(prestart, mergedscene.start - margin)
    mend = min(mergedscene.end + margin, video_total_length)
    clips.append(clip.subclip(mstart, mend))
    print(f" mstart = {str(mstart)}   mend = {str(mend)}")
    pos += 1 + ascene.repeatCount
    prestart = mend

#add last scene if not added yet
apos = len(merged_scenes)-1
#if set to False, we picke second-last one
found = False
while apos >= 0:
    mscene = merged_scenes[apos]
    if len(mscene.mergedContent) == 0 or (mscene.persons == 0 and len(mscene.mergedObjects) == 0):
        apos -= 1
        continue
    if found:
        break
    found = True 
    apos -= 1
if mscene.start > prestart:
    clips.append(clip.subclip(mscene.start, min(mscene.start + 10, mscene.end) ))

finalvideo = concatenate_videoclips(clips)
finalvideo.write_videofile(args.filename + "-" +
                           str(total_output_time) + "-out.mp4")

# output climax
climax_time = args.climax
if climax_time <= 5:
    climax_time = 180  # 3 minutes
for scene in scenes:
    if scene.climax:
        mergedscene = merged_scenes[scene.bestDialogOrder]
        clips = []
        margin = (climax_time - mergedscene.end + mergedscene.start)/2
        mstart = max(0, mergedscene.start - margin)
        mend = min(video_total_length, mergedscene.end + margin)
        clips.append(clip.subclip(mstart, mend))
        finalvideo = concatenate_videoclips(clips)
        finalvideo.write_videofile(args.filename + "-" +
                                   str(climax_time) + '-climax' + "-out.mp4")
