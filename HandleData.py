import os
import argparse
import sys
import pandas as pd

parser = argparse.ArgumentParser(description='combines files into a pickle object, processes the data, and divides into training sets')
parser.add_argument('-i', dest='folder', default="/methylation_data", type=str,
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

if args.test:
    print("Running in test mode: Downloading real 450K methylation data...")
    import GEOparse
    import pandas as pd

    # Download GEO dataset (example: GSE68777)
    gse = GEOparse.get_GEO("GSE68777", destdir="./")
    print("b")
    # Extract beta values (methylation data)
    df = gse.pivot_samples("VALUE")
    print("a")
    # Save for later use
    df.to_csv("test_450K_methylation.csv")

    # Reload with explicit dtype settings
    df = pd.read_csv("test_450K_methylation.csv", dtype=str, low_memory=False)

    print("Processed methylation data:")
    print(df.head())
