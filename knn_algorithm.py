import csv 
import random
from scipy.spatial import distance
import operator
import math

# with open('C:/Users/White_Devil/Desktop/machine learning practise/iris.data','rt') as cf:
# 	line = csv.reader(cf)
# 	for row in line:
# 		print(' ,'.join(row))

def loadData(filename, split,trainingSet=[],testSet=[]):
	with open(filename, 'r') as csvfile:
		lines=csv.reader(csvfile)
		dataset=list(lines)
		for x in range(len(dataset)-1):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
			if random.random()<split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])					


def EuclideanDistance(p1,p2,length):
	distances=0
	for x in range(length):
		distances+=pow((p1[x]-p2[x]),2)
	return math.sqrt(distances)	

def getNeighbors(trainingSet,testInstance,k):
	distance =[]
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist=EuclideanDistance(testInstance,trainingSet[x],length)
		distance.append((trainingSet[x],dist))
	distance.sort(key=operator.itemgetter(1))
	neighbors=[]
	for x in range(k):
		neighbors.append(distance[x][0])
	return neighbors		

# trainingSet=[[2,2,2,'a'],[4,4,4,'b']]
# testInstance=[5,5,5]
# k=1
# neighbors=getNeighbors(trainingSet,testInstance,1)
# print(neighbors)	

def getResponse(neighbors):
	classVotes={}
	for x in range (len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response]+=1
		else:
			classVotes[response]=1	
	sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
	return sortedVotes[0][0]		

# neighbors=[[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
# response = getResponse(neighbors)
# print(response)	

def getAccuracy(testSet,predictions):
	correct=0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct+=1
	return (correct/float(len(testSet)))*100.0
	

# testSet =[[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
# predictions =['a','a','a']
# accuracy = getAccuracy(testSet,predictions)
# print(accuracy)			

def main():
	trainingSet=[]
	testSet=[]
	split=0.67
	loadData(r'iris.data',split,trainingSet,testSet)
	print("Train : " + repr(len(trainingSet)))	
	print("Test : " + repr(len(testSet)))	

	predictions=[]
	k=3
	for x in range(len(testSet)):
		neighbors=getNeighbors(trainingSet,testSet[x],k)
		result=getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) +' , actual=' + repr(testSet[x][-1]))
		accuracy=getAccuracy(testSet,predictions)
		print('Accuracy: ' + repr(accuracy) + '%')


main()		