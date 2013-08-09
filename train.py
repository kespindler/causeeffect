import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation as cval
from sklearn.pipeline import Pipeline
import pandas as pd

def feature_extractor():
    features = [('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
                #('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.count_unique)),
                #('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.count_unique)),
                
                ('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.percentage_unique)),
                ('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.percentage_unique)),

                ('A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)),
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)),
                #('Conditional A on B', ['A', 'A type', 'B', 'B type'], f.MultiColumnTransform(f.conditional_info)),
                #('Conditional B on A', ['B', 'B type', 'A', 'A type'], f.MultiColumnTransform(f.conditional_info)),
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference))]
    combined = f.FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=50, 
                                                verbose=2,
                                                n_jobs=1,
                                                min_samples_split=10,
                                                random_state=1))]
    return Pipeline(steps)

if __name__=="__main__":
    print("Reading in the training data")
    train_raw = data_io.read_train_pairs()
    target = data_io.read_train_target()
    info = data_io.read_train_info()

    train = train_raw.join(info)

    #max_categoriesA = max(len(np.unique(x)) for x in 
    #    train['A'][info['A type'] == 'Categorical'])
    #max_categoriesB = max(len(np.unique(x)) for x in 
    #    train['B'][info['B type'] == 'Categorical'])
    #max_categories = max([max_categoriesA, max_categoriesB])

    classifier = get_pipeline()
    folds = cval.KFold(len(train), n_folds=2, indices=False)

    results = []
    for i, fold in enumerate(folds):
        print("Extracting features and training model for fold " + str(i))
        traincv, testcv = fold
        classifier.fit(train[traincv], target[traincv])
        results.append(classifier.score(train[testcv], target[testcv]))

    print(results)
    print('Score: ' + str(np.array(results).mean()))

    #classifier.fit(train, target.Target)

    print("Saving the classifier")
    data_io.save_model(classifier)
  
