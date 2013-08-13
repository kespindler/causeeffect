import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation as cval
from sklearn.pipeline import Pipeline
import pandas as pd
import sys

def feature_extractor():
    features = [('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
                ('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.count_unique)),
                ('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.count_unique)),
                # ('A: Number of Unique Samples', 'A', f.SimpleTransform()),

                ('A: Ratio of Unique Samples', 'A', f.SimpleTransform(transformer=f.percentage_unique)),
                ('B: Ratio of Unique Samples', 'B', f.SimpleTransform(transformer=f.percentage_unique)),

                ('A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)),
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)),
                ('A: Injectivity into B', ['A','B'], f.MultiColumnTransform(f.injectivity)),
                ('B: Injectivity into A', ['B','A'], f.MultiColumnTransform(f.injectivity)),

                ('Conditional A on B', ['A', 'A type', 'B', 'B type'], f.MultiColumnTransform(f.conditional_info)),
                ('Conditional B on A', ['B', 'B type', 'A', 'A type'], f.MultiColumnTransform(f.conditional_info)),
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference))]

    combined = f.FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=50, 
                                                verbose=2,
                                                n_jobs=2,
                                                min_samples_split=10,
                                                random_state=1,
                                                compute_importances=True))]
    return Pipeline(steps)

if __name__=="__main__":

    print("Reading in the training data")
    train_raw = data_io.read_train_pairs()
    target = data_io.read_train_target()
    info = data_io.read_train_info()

    train = train_raw.join(info)

    classifier = get_pipeline()

### FOLDS CODE
    folds = cval.KFold(len(train), n_folds=2, indices=False)

   
    results = []
    for i, fold in enumerate(folds):
        print("Extracting features and training model for fold " + str(i))
        traincv, testcv = fold
        classifier.fit(train[traincv], target[traincv])
        results.append(classifier.score(train[testcv], target[testcv]))

    print(results)
    print('Score: ' + str(np.array(results).mean()))
###  REGULAR RUN
#
#    #classifier.fit(train, target.Target)
#
#    print("Saving the classifier")
#    data_io.save_model(classifier)
#  
