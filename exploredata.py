import csv
import json
import numpy as np
import os
import pandas as pd
import pickle

import features as f


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation as cval
from sklearn.pipeline import Pipeline

import sys
import matplotlib.pyplot as plt
import data_io

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df

def read_train_pairs():
    train_path = get_paths()["train_pairs_path"]
    print pd.read_csv(train_path, index_col="SampleID")
    return parse_dataframe(pd.read_csv(train_path, index_col="SampleID"))
def showData(i,reverse=False):
	Avals = pd.Series(train_raw.values[i][0])
	Bvals = pd.Series(train_raw.values[i][1])
	if reverse:
		Avals, Bvals = Bvals, Avals
	newD = pd.DataFrame({'A':Avals,'B':Bvals})
	
	plt.scatter(newD.A,newD.B)
	plt.show()

if __name__=="__main__":
	print("Reading in the training data")
	train_raw = read_train_pairs()
	