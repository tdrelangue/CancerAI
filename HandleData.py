import os
import argparse
import sys
import pandas as pd
from random import seed, shuffle
import numpy as np

from Dataset import Dataset

parser = argparse.ArgumentParser(description='combines files into a pickle object, processes the data, and divides into training sets')
parser.add_argument('-i', dest='folder', default="methylation_data", type=str,
                    help='Folder containing the files to be processed. (Marcin can enter b, p, or k for defaults)')
parser.add_argument('-s', dest='sets', default="0.7:0.2", type=str,
                    help='Controls how the training, validation, and testing sets are divided. Expects text in the form a:b:c or a:b where a is the training set, b is validation and c is testing. If testing is omitted then it is assumed to be 1-(a+b)')
parser.add_argument('-r', dest='seed', default=None, type=int,
                    help='Set the random seed')
parser.add_argument('-p', dest='param_copy', action="store_true",
                    help='If provided then the program will load the parameters from the file provided by -n. Overrides -d, -t, -a, and -b')
parser.add_argument('-t', dest='useTree', action="store_true",
                    help='If provided then the program will decrease the dimensions using the decision tree instead of PCA.')
parser.add_argument('-d', dest='dimensions', default=2, type=int,
                    help='Controls how many dimensions to keep after PCA or decisionTree.')
parser.add_argument('-a', dest='minPatientNA', default=0.15, type=float,
                    help='Sets the maximum proportion of NA values a patient can have before being discarded. Overridden if -p is set')
parser.add_argument('-b', dest='minPositionNA', default=0.15, type=float,
                    help='Sets the maximum proportion of NA values a cpg site can have before being discarded. Overridden if -p is set')
parser.add_argument('-o', dest='patientsFile', default=None, type=str,
                    help='Specify the location to output the patient file to. If none will create it in the same folder as the input with a generated name.')
parser.add_argument('-n', dest='paramsFile', default=None, type=str,
                    help='Specify the location to output the parameter file to. If none will create it in the same folder as the input with a generated name.')
parser.add_argument("--test", action="store_true", help="Enable test mode with synthetic data")

args = parser.parse_args()

inputFolder = args.folder
if args.test:
    inputFolder = "methylation_data/test/"
    os.makedirs(os.path.dirname(inputFolder), exist_ok=True)
elif(inputFolder == "b"):
    inputFolder = "methylation_data/bladder/"
elif(inputFolder == "p"):
    inputFolder = "methylation_data/prostate/"
elif(inputFolder == "k"):
    inputFolder = "methylation_data/kidney/"
if(inputFolder[-1] != "/"):
    inputFolder += "/"
if(not os.path.exists(inputFolder)):
    print("Error! No folder exists at ", inputFolder)
    print("Exiting")
    sys.exit(1)
folderHead, folderTail = os.path.split(inputFolder[:-1])

sets = args.sets.split(":")
trainingRatio = float(sets[0])
validationRatio = float(sets[1])
if(len(sets)>2):
    ratio_sum = trainingRatio + validationRatio + float(sets[2])
    trainingRatio = trainingRatio/ratio_sum
    validationRatio = validationRatio/ratio_sum
testingRatio = 1 - trainingRatio - validationRatio

seed(args.seed)

recreatePCA = not args.param_copy
if(recreatePCA):
    pcaDimensions = args.dimensions
    minPatientNA = args.minPatientNA
    minPositionNA = args.minPositionNA

patientsFile = args.patientsFile
if(patientsFile == None):
    if(recreatePCA):
        if(pcaDimensions > 0):
            if args.useTree:
                patientsFile = os.path.join(inputFolder, folderTail+"_TOP"+str(pcaDimensions)+".pkl")
            else:
                patientsFile = os.path.join(inputFolder, folderTail+"_PCA"+str(pcaDimensions)+".pkl")
        else:
            patientsFile = os.path.join(inputFolder, folderTail+"_NoPCA.pkl")
    else:
        patientsFile = os.path.join(inputFolder, folderTail+"_PCAcopied.pkl")
        
paramsFile = args.paramsFile


def loadBetaValues(folder):
    subdirs = os.listdir(folder)
    
    patients = []
    #maxFiles = 6
    i = 0
    for currentDirectory in subdirs:
        i += 1
        #if(i >= maxFiles):
        #    patients[0]["Cancer"] = True
        #    patients[1]["Cancer"] = True
        #    patients[2]["Cancer"] = True
        #    patients[3]["Cancer"] = False
        #    patients[4]["Cancer"] = False
        #    break
        
        #try:
        betaDict = {}
        try:
            files = os.listdir(os.path.join(folder, currentDirectory))
        except:
            print("skipping directory", currentDirectory,"as it could not be opened")
            continue
            
        print("reading directory", currentDirectory, "Found", len(files), "files")
        
        for currentFileName in files:
            if(currentFileName[:3] == "jhu" and "HumanMethylation450" in currentFileName and currentFileName[-4:] == ".txt"):
                with open(os.path.join(folder, currentDirectory, currentFileName)) as currentFile:
                    #print("dir", i, " file is open. Reading lines")
                    j = 0
                    for line in currentFile:
                        #print("reading line", j, "of file in dir", i)
                        try:
                            separated = line.split("\t")
                            location = separated[2].strip() + ':' + separated[3].strip()
                            try:
                                betaDict[location] = float(separated[1].strip())
                            except ValueError:
                                #any value that can not be interpreted as a float is set to NA
                                betaDict[location] = Dataset.NAValue
                        except KeyError as e:
                            print("exception line", j, "of file in dir", i)
                            print(str(e))
                            print("checking for", separated[0].strip())
                            pass
                        j+=1
                            #skip any lines that can't be matched (presumably only the header
            
            #jhu-usc.edu_PRAD.HumanMethylation450.18.lvl-3.TCGA-ZG-A9LZ-01A-11D-A41L-05.gdc_hg38.txt
            #                                                           ^^ tissue type identifier
            if(len(betaDict) > 0):#skip the folders that don't use 450k probes
                try:
                    if(currentFileName.split("-")[5][:2] == "01"):
                        betaDict["Cancer"] = True
                        patients.append(betaDict)
                    elif(currentFileName.split("-")[5][:2] == "11"):
                        betaDict["Cancer"] = False
                        patients.append(betaDict)
                
                    #print("adding new patient. Current length =", len(patients))
                except IndexError:
                    pass
    
    
    #Catch weird cancer key error
    for patientNum in range(0, len(patients)):
        for key in patients[patientNum - 1].keys():
            if(not key in patients[patientNum].keys()):
                print("1: ERROR patient", patientNum - 1, "has key", key, "but patient", patientNum, "does not!!!")
    
    
    return patients

if __name__ == "__main__":
    if args.test:
        if os.path.exists("methylation_data/test/test_PCA2.pkl"):
            td = Dataset(pklLocation="methylation_data/test/test_PCA2.pkl",trainingRatio=trainingRatio, validationRatio=validationRatio, seedValue=args.seed, pca_params=None)
        else:
            td = Dataset(test_mode=True, seedValue=args.seed, trainingRatio=trainingRatio, validationRatio=validationRatio, pca_params=paramsFile)
        td.save("methylation_data/test/test_PCA2.pkl")
        print(td)
    else:
        unprocessedPatientData = loadBetaValues(inputFolder)