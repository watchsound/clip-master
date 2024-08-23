from typing import List
from abc import ABC, abstractmethod
import numpy as np

class HasTimeRange(ABC):

    @abstractmethod
    def getStart(self):
        pass

    @abstractmethod
    def getEnd(self):
        pass


class TimeRangeUtils:
    @staticmethod
    def overlap(r1, r2, margin):
        s1 = r1.getStart() - margin
        e1 = r1.getEnd() + margin
        s2 = r2.getStart()
        e2 = r2.getEnd()

        return max(s1, s2) < min(e1, e2)


class SRT(HasTimeRange):

    def __init__(self):
        self.order = 0
        self.start = 0.0
        self.end = 0.0
        self.optional = False
        self.content = ""

    def getOrder(self):
        return self.order

    def setOrder(self, order):
        self.order = order

    def getStart(self):
        return self.start

    def setStart(self, start):
        self.start = start

    def getEnd(self):
        return self.end

    def setEnd(self, end):
        self.end = end

    def isOptional(self):
        return self.optional

    def setOptional(self, optional):
        self.optional = optional

    def getContent(self):
        return self.content

    def setContent(self, content):
        self.content = content


class VideoScene(HasTimeRange):

    def __init__(self):
        self.order = 0
        self.start = 0.0
        self.end = 0.0
        self.objects = ""
        self.persons = 0
        self.sexValue = 0.0

    def getOrder(self):
        return self.order

    def setOrder(self, order):
        self.order = order

    def getStart(self):
        return self.start

    def setStart(self, start):
        self.start = start

    def getEnd(self):
        return self.end

    def setEnd(self, end):
        self.end = end

    def getObjects(self):
        return self.objects

    def setObjects(self, objects):
        self.objects = objects

    def getSexValue(self):
        return self.sexValue

    def setSexValue(self, sexValue):
        self.sexValue = sexValue


class MergedScene(HasTimeRange):

    def __init__(self):
        self.order = 0
        self.start = 1111111
        self.end = -1
        self.srtOrderStart = 1111111
        self.srtOrderEnd = -1
        self.sceneOrderStart = 1111111
        self.sceneOrderEnd = -1
        self.mergedContent = ""
        self.mergedObjects = ""
        self.persons = 0
        self.mergedSex = 0.0
        self.sentiments = 0.0

    def getOrder(self):
        return self.order

    def setOrder(self, order):
        self.order = order

    def getStart(self):
        return self.start

    def setStart(self, start):
        self.start = start

    def getEnd(self):
        return self.end

    def setEnd(self, end):
        self.end = end

    def getSrtOrderStart(self):
        return self.srtOrderStart

    def setSrtOrderStart(self, srtOrderStart):
        self.srtOrderStart = srtOrderStart

    def getSrtOrderEnd(self):
        return self.srtOrderEnd

    def setSrtOrderEnd(self, srtOrderEnd):
        self.srtOrderEnd = srtOrderEnd

    def getSceneOrderStart(self):
        return self.sceneOrderStart

    def setSceneOrderStart(self, sceneOrderStart):
        self.sceneOrderStart = sceneOrderStart

    def getSceneOrderEnd(self):
        return self.sceneOrderEnd

    def setSceneOrderEnd(self, sceneOrderEnd):
        self.sceneOrderEnd = sceneOrderEnd

    def getMergedContent(self):
        return self.mergedContent

    def setMergedContent(self, mergedContent):
        self.mergedContent = mergedContent

    def getMergedObjects(self):
        return self.mergedObjects

    def setMergedObjects(self, mergedObjects):
        self.mergedObjects = mergedObjects

    def getSentiments(self):
        return self.sentiments

    def setSentiments(self, sentiments):
        self.sentiments = sentiments

    def merge(self, scene):
        self.start = min(self.start, scene.start) 
        self.end = max(self.end, scene.end) 

        self.sceneOrderStart = min(self.sceneOrderStart, scene.order)
        self.sceneOrderEnd = max(self.sceneOrderEnd, scene.order)
 
        if self.mergedObjects is None:
            self.mergedObjects = scene.objects
        else:
            self.mergedObjects += " " + scene.objects
        
        if self.mergedSex is None:
            self.mergedSex = scene.sexValue
        else:
            self.mergedSex = max(self.mergedSex, scene.sexValue)
        
        self.persons = max(self.persons, scene.persons)

        

    def mergeSRT(self, srt):
        self.start = min(self.start, srt.start)
        self.end = max(self.end, srt.end)
 
        self.srtOrderStart = min(self.srtOrderStart, srt.order)
        self.srtOrderEnd = max(self.srtOrderEnd, srt.order)
  
        if srt.optional is False:
            if self.mergedContent is None:
                self.mergedContent = srt.content
            else:
                self.mergedContent += " " + srt.content


class ContextScene:
    def __init__(self):
        self.order = 0
        self.summary = ""
        self.description = ""
        self.climax = False
        self.matchedDialogOrders = []
        self.matchedDialogScore = [1.5,1,0.5]
        self.bestDialogOrder = -1
        self.repeatCount = 0
        self.suggestObjects = ""

    def getStart(self):
        return self.start

    def setStart(self, start):
        self.start = start

    def getEnd(self):
        return self.end

    def setEnd(self, end):
        self.end = end

    
    @staticmethod
    def dyBestPath(sceneList: List['ContextScene']) -> int:
        bestLength = 0
        bestPenalty = 0

        for p in np.arange(0, 3, 0.25):
            length = ContextScene.dyBestPathWithPenalty(sceneList, -p)
            if length > bestLength:
                bestLength = length
                bestPenalty = -p

        return ContextScene.dyBestPathWithPenalty(sceneList, bestPenalty)

    @staticmethod
    def dyBestPathWithPenalty(sceneList: List['ContextScene'], penalty: float) -> int:
        count = len(sceneList)
        values = [[0] * 4 for _ in range(count)]
        dir = [[0] * 4 for _ in range(count)]

       

        cs = sceneList[0]
        values[0][0] = cs.matchedDialogScore[0]
        values[0][1] = cs.matchedDialogScore[1]
        values[0][2] = cs.matchedDialogScore[2]
        values[0][3] = penalty

        dir[0][0] = 0
        dir[0][1] = 0
        dir[1][2] = 0
        dir[1][3] = 0

        for i in range(1, count):
            cs = sceneList[i]

            for j in range(3):
                backSteps = 0
                for k in range(i - 1, -1, -1):
                    if dir[k][j] != 3:
                        break
                    backSteps += 1

                adjstPrevPos = i - 1 - backSteps
                ps = sceneList[adjstPrevPos]

              
                penalties = backSteps * penalty

                value = float('-inf')
                if cs.matchedDialogOrders[j] >= ps.matchedDialogOrders[0]:
                    value = values[adjstPrevPos][0] + cs.matchedDialogScore[j] + penalties
                    dir[i][j] = 0
                if cs.matchedDialogOrders[j] >= ps.matchedDialogOrders[1]:
                    if value < values[adjstPrevPos][1] + cs.matchedDialogScore[j] + penalties:
                        value = values[adjstPrevPos][1] + \
                            cs.matchedDialogScore[j] + penalties
                        dir[i][j] = 1
                if cs.matchedDialogOrders[j] >= ps.matchedDialogOrders[2]:
                    if value < values[adjstPrevPos][2] + cs.matchedDialogScore[j] + penalties:
                        value = values[adjstPrevPos][2] + \
                            cs.matchedDialogScore[j] + penalties
                        dir[i][j] = 2
                if value == float('-inf'):
                    dir[i][j] = 3

                values[i][j] = value

        choose = 0
        cs = sceneList[count - 1]
        if values[count - 1][0] >= values[count - 1][1] and values[count - 1][0] >= values[count - 1][2]:
            cs.bestDialogOrder = cs.matchedDialogOrders[0]
            choose = dir[count - 1][0]
        elif values[count - 1][1] >= values[count - 1][0] and values[count - 1][1] >= values[count - 1][2]:
            cs.bestDialogOrder = cs.matchedDialogOrders[1]
            choose = dir[count - 1][1]
        else:
            cs.bestDialogOrder = cs.matchedDialogOrders[2]
            choose = dir[count - 1][2]

        pathLength = 1
        for i in range(count - 2, -1, -1):
            cs = sceneList[i]
            if dir[i][choose] == 3:
                cs.bestDialogOrder = -1
                continue
            cs.bestDialogOrder = cs.matchedDialogOrders[choose]
            pathLength += 1
            choose = dir[i][choose]

        return pathLength
