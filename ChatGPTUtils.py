from VideoComponents import SRT
from VideoComponents import VideoScene
from VideoComponents import MergedScene
from VideoComponents import TimeRangeUtils
from VideoComponents import ContextScene

def getContextSceneBy(gptResponse):
    css = []

    sceneTag = "Scene:"
    objTag = "Objects:"
    climxTag = "Climax:"
    descTag = "Description:"

    lines = gptResponse.split("\n")
    curScene = None
    for i in range(len(lines)):
        if lines[i].strip() == "":
            continue
        if lines[i].startswith(sceneTag):
            curScene = ContextScene()
            css.append(curScene)
            curScene.order = len(css)
            curScene.summary = lines[i][len(sceneTag):]
            continue

        if lines[i].startswith(objTag):
            curScene.suggestObjects = lines[i][len(objTag):]
            continue
        if lines[i].startswith(climxTag):
            curScene.climax = lines[i][len(climxTag):].strip().lower() == "yes"
            continue

        if lines[i].startswith(descTag):
            curScene.description = lines[i]
            continue
        curScene.description += lines[i]

    return css
