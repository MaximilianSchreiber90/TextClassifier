from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
   
'''
df is a dataframe
label_list is a list containing all unique labels contained in the dataframe          # calculate yourself from df
top_i the number of top features to be returned 
returns the most important features per label (according to binary SGDClassifier)
'''
def most_important_SGD_binary(df, label_list, i):           # TODO add var for % top features
    feature_map = {}
    for label in label_list:
        #getting a testset
        df_temp = get_test_data_for_label(df.copy(), label, 0.3)
        df_list.append(df_temp)

        #splitting data
        split_map_all = split_dataset(df_temp)

        X_train_all = split_map_all['X_train']
        y_train = split_map_all['y_train']
        X_test_all = split_map_all['X_test']
        y_test = split_map_all['y_test']

        tfidf_vectorizer =  TfidfVectorizer(stop_words='english', max_df=0.25)

        X_train = tfidf_vectorizer.fit_transform(X_train_all)
        X_test = tfidf_vectorizer.transform(X_test_all)

        feature_names = tfidf_vectorizer.get_feature_names()

        clf = SGDClassifier(loss='log', penalty='l1', l1_ratio=0.9, learning_rate='optimal', n_iter=10, shuffle=False, n_jobs=3, fit_intercept=True)
        clf.fit(X_train, y_train)
        for i in range(0, clf.coef_.shape[0]):
            top_feature_names = []
            top_indices = np.argsort(clf.coef_[i])[-top_i:]
            for j in top50_indices:
                top_feature_names.add(feature_names[j])
            feature_map[label] = top_feature_names
            
        return feature_map
                
#------------------------------------------------------------------------------#                
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
'''
df is a dataframe
label_list is a list containing all unique labels contained in the dataframe          # calculate yourself from df
top_i the number of top features to be returned 
returns the most important features per label (according to binary SGDClassifier)
'''
def most_important_SGD_multi_all(df, top_i):
    #splitting data
    split_map_all = split_dataset(df)

    X_train_all = split_map_all['X_train']
    y_train_all = split_map_all['y_train']

    X_test_all = split_map_all['X_test']
    y_test_all = split_map_all['y_test']

    #converting label
    convert_label_map = convert_labels_for_multilabeling(y_train_all, y_test_all)
    y_train = convert_label_map['y_train_encoded']
    y_test = convert_label_map['y_test_encoded']
    tfidf_vectorizer =  TfidfVectorizer(stop_words='english', max_df=0.25)

    X_train = tfidf_vectorizer.fit_transform(X_train_all)
    X_test = tfidf_vectorizer.transform(X_test_all)

    feature_names_SGD = tfidf_vectorizer.get_feature_names()
        
    clf = OneVsRestClassifier(SGDClassifier(
        loss='log', penalty='l1', 
        l1_ratio=0.7, learning_rate='optimal', 
        n_iter=50, shuffle=False, n_jobs=3, 
        fit_intercept=True))

    clf.fit(X__train, y__train)
    
    #iterates and outputs the most important input, not feature
    feature_map = {}
    for i in range(0, clf.coef_.shape[0]):
        top50_indices = np.argsort(clf.coef_[i])[top_i:]
        top_feature_names = []
        for j in top50_indices:
            top_feature_names.add(feature_names[j])
        feature_map[label] = top_feature_names
    
    return feature_map

########################################################################################################################################################
#################################################### Features Random Forest ############################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt

'''
df is a dataframe
label_list is a list containing all unique labels contained in the dataframe          # calculate yourself from df
top_i the number of top features to be returned 
returns the most important features per label (according to binary SGDClassifier)
'''
def most_important_SGD_multi(df, label_list, i):
    feature_forest_map = {}
    for label in label_list:
        print('Comparing data for label ',label)
        df_temp = get_test_data_for_label(df.copy(), label, 0.3)
        df_list.append(df_temp)

        #splitting data
        split_map_all = split_dataset(df_temp)

        X_train_all = split_map_all['X_train']
        y_train = split_map_all['y_train']

        X_test_all = split_map_all['X_test']
        y_test = split_map_all['y_test']

        tfidf_vectorizer =  TfidfVectorizer(stop_words='english', max_df=0.25)

        X_train = tfidf_vectorizer.fit_transform(X_train_all)
        X_test = tfidf_vectorizer.transform(X_test_all)

        X_train_dense = X_train[:10000].copy().toarray()
        X_test_dense = X_test[:10000].copy().toarray()
        y_train_dense = y_train[:10000].copy()
        y_test_dense = y_test[:10000].copy()

        forest = RandomForestClassifier(n_estimators=50, min_samples_split=0.01, class_weight={label:90, "NONE":10})
        forest.fit(X_train_dense, y_train_dense)

        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        feature_list_tree = []
        feature_names = tfidf_vectorizer.get_feature_names()

        for f in range(X_train_dense.shape[1]):
            feature_list_tree.append(feature_names[indices[f]])
    #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        feature_forest_map[label] = feature_list_tree    

        
'''
label_list_all = ['SCIENCE', 'SPORTS', 'POLITICS', 'LIFESTYLE', 'HEALTH', 'EDUCATION', 'WORLDPOST', 'IMPACT', 'BUSINESS', 'RANDOM']
label_list = ['SCIENCE', 'POLITICS', 'BUSINESS']
#df, labelname, split_ratio
df_science = get_test_data_for_label(df_all.copy(), 'SCIENCE', 0.3)
df_sport = get_test_data_for_label(df_all.copy(), 'SPORT', 0.3)
df_politics = get_test_data_for_label(df_all.copy(), 'POLITICS', 0.3)
df_lifestyle = get_test_data_for_label(df_all.copy(), 'LIFESTYLE', 0.3)
df_health = get_test_data_for_label(df_all.copy(), 'HEALTH', 0.3)
df_education = get_test_data_for_label(df_all.copy(), 'EDUCATION', 0.3)
df_worldpost = get_test_data_for_label(df_all.copy(), 'WORLDPOST', 0.3)
df_impact = get_test_data_for_label(df_all.copy(), 'IMPACT', 0.3)
df_business = get_test_data_for_label(df_all.copy(), 'BUSINESS', 0.3)
df_random = get_test_data_for_label(df_all.copy(), 'RANDOM', 0.3)
'''

########################################################################################################################################################
#################################################### SVM ###############################################################################################
