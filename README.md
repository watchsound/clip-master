# intro
The project involves taking a full-length movie video as input and generating a concise 2-5 minute highlight reel or clip from it. The goal is to automatically identify and extract key scenes that capture the essence of the movie, providing a brief yet compelling summary of the content.

The process may involve analyzing the movie's audio, visual elements, and narrative structure to determine the most impactful moments. These selected scenes are then compiled into a short, cohesive video clip that represents the movie's storyline, major plot points, or most visually striking sequences.

This tool could be useful for creating promotional material, summarizing movies for viewers, or generating quick previews for content recommendations.

# implementation

//step1: download video 
//step2: if video does not contains srt file, download srt file
//step3: get abstraction
//step4: use chatgpt to get scenes from abstraction
//step5: decompose video into v-scenes 
//step6: add object to v-scene images
//step7: from v-scenes get cluster dialogs
//step7.5  use srt to adjust v-scenes (merge v-scenes if belong to one sentence in srt) 
//step8: ask chatgpt to find best match cluster for each scene
//step9: add sentiment score 
//step9.5:  add distance between objects and scene.
//step10: alignment.
//step11: create video