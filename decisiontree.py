import sys
import csv
import random
import numpy as np

# to print decision tree as a string
decstring = None

# read input file
def readfile(inputFileName):
	tdata = []
	attributes = []
	cset = []
	tcset = []
	reader = csv.reader(open(inputFileName, "rb"), delimiter='\t')
	for row, record in enumerate(reader):
		if row == 0:
			#print record
			attributes = record
			attributes.remove('CLASS')
		else:
			dic = {}
			#print attributes
			#print record
			for i in range(len(attributes)):
				if 'true' in record[i]:
					dic[attributes[i]] = True
				else:
					dic[attributes[i]] = False
			tdata.append(dic)
			if 'true' in record[len(attributes)]:
				cset.append(True)
				tcset.append([dic, True])
			else:
				cset.append(False)
				tcset.append([dic, False])	
 
	return attributes, tcset, tdata, cset

# divide data into training data and testdata
def divtraintestsets(tcset):
	tsa = range(len(tcset))
 	random.shuffle(tsa)
 	tsa = [tsa[i] for i in range(trainingSetSize)]
 	trset = []
 	testset = []
 	for i in range(len(tcset)):
 		if i in tsa:
 			trset.append(tcset[i])
 		else:
 			testset.append(tcset[i])
 	return trset, testset				 

# calculate probability of positive in the dataset
def posprob(tcset):
	trnum = 0
	for t in tcset:
		if t[1] == True:
			trnum = trnum + 1
		
	return float(trnum)/len(tcset)		

# Separate a data set into subsets for values true and false
def septrset(trset, attr):
	ctrset = []
	cflset = []
	for t in trset:
		if t[0][attr] == True:
			ctrset.append(t)
		else:
			cflset.append(t)
	return ctrset, cflset			 

# Calculate entroty
def entropy(tcset):
	ptr = posprob(tcset)
	if ptr == 1 or ptr ==0:
		return 0
	pfl = 1 - ptr
	etr = -ptr * np.log2(ptr)
	efl = -pfl * np.log2(pfl)
	return etr + efl


# Calculate information gain
def infogain(tcset, attr):
	ctrset, cflset = septrset(tcset, attr)
	gtr = len(ctrset)/len(tcset) * entropy(ctrset)
	gfl = len(cflset)/len(tcset) * entropy(cflset)
	return entropy(tcset) - gtr - gfl

# Select best attribute for splitting
def bestattr(tcset, attributes):
	bestattr = None
	linfgain = 0
	for attr in attributes:
		infgain = infogain(tcset, attr)
		if linfgain < infgain:
			bestattr = attr
			linfgain = infgain

	return bestattr		


# ID3 decision tree (return as a list)
def ID3(trset, attributes, pattr):
	tprob = posprob(trset)
	tstree = None 
	fstree = None
	global decstring
	if tprob is 1:
		decstring =  decstring + '\nparent: ' + pattr + '+'
		return True
	elif tprob is 0:
		decstring = decstring + '\nparent: ' + pattr + '-'
		return False	
	elif len(attributes) == 0:
		if tprob >= 0.5:
			decstring = decstring + '\nparent: '+ pattr+ '+'
			return True
		else:
			decstring = decstring + '\nparent: '+ pattr + '-'
			return False
	else:
		attr = bestattr(tcset, attributes)
		attributes.remove(attr)
		ctrset, cflset = septrset(trset, attr)
		if len(ctrset) == 0:
			if tprob >= 0.5:
				tstree = [attr, True, True]
			else:
				tstree = [attr, True, False]
		else:
			tstree = [attr, True, ID3(ctrset, attributes, attr)]
		if len(cflset) == 0:
			if tprob >= 0.5:
				fstree = [attr, False,True]
			else:
				fstree = [attr, False, False]
		else:
			fstree = [attr, False, ID3(cflset, attributes, attr)]
	
	if type(tstree[2]) is bool:
		 tchild = 'leaf'
	else:
		tchild = tstree[2][0][0]
	if type(fstree[2]) is bool:
		fchild = 'leaf'
	else:
		fchild = fstree[2][0][0]			 

	decstring = decstring + '\nparent: ' + pattr + ' attribute: ' + attr+ 'trueChild:'+ tchild + 'falseChild:' + fchild		
	return tstree, fstree		 

# Test one data point using decision tree
def test(dectree, td):
	attr = dectree[0][0]
	attrv = td[attr]
	subtree = None
	if dectree[0][1] == attrv:
		subtree = dectree[0][2]
	else:
		subtree = dectree[1][2]

	if subtree == True or subtree == False:
		return subtree
	else:
		return test(subtree, td)	
 
# Classify using ID3
def classifyID3(dectree, testset):
	correct = 0.0
	for t in testset:
		res = test(dectree, t[0])
		if res == t[1]:
			correct = correct + 1
	return correct/len(testset)	

if __name__ == '__main__':
	try:
		inputFileName = sys.argv[1]
		trainingSetSize = int(sys.argv[2])
		numberOfTrials = int(sys.argv[3])
		verbose = sys.argv[4]
	except:
		print "Usage: python decisiontree.py <inputFileName> <trainingSetSize> numberOfTrials> <verbose>"
		sys.exit(0)
 
 	### Step 1
 	sumdc = 0
 	sumpp = 0
 	for i in range(numberOfTrials):
 		### Step 1 read data
 		attributes, tcset, tset, cset = readfile(inputFileName)
 	
 		print "TRIAL NUMBER: ", i
 		print "----------------------\n"
 		### Step 2 divide data into training data and test data
 		trset, testset = divtraintestsets(tcset)
 	
 		### Step 3 prior prob using training data
 		expprob = posprob(trset)
 	
 		### Step 4 Construct a decision tree
 		decstring = ''
 		dectree = ID3(trset, attributes, 'root')
 		decstring = decstring.split('\n')
 		decstring.reverse()
 		decstring = '\n'.join(decstring)
 		print "DECISION TREE STRUCTURE:"
 		print decstring
 	 

 		
 		#### Classify the examples in test set using ID3
 		testres = classifyID3(dectree, testset)
 		accperct = testres * 100
 		sumdc = sumdc + accperct
 		print "\t\tPercent of test cases correctly classified by a decision tree build with ID3 = ", accperct
  		
  		### Classify by selecting the most likely class
 		
 		actprob = posprob(testset)
 		priorprob = expprob
 		accuracy = 1 - 2 * abs(actprob-priorprob)
 		acctperct = accuracy * 100
 		sumpp = sumpp + acctperct
 		print "\t\tPercent of test cases correctly classified by using prior probabilities from the training example set = ", acctperct
  		print "\n"

 	print "\nexample file used =", inputFileName
 	print "number of trials =", numberOfTrials
 	print "training set size for each trial =", trainingSetSize
 	print "mean performance of decision tree over all trials = ", sumdc/numberOfTrials, " correct classification"
	print "mean performance of using prior probability derived from the training set = ", sumpp/numberOfTrials, " correct classification"



