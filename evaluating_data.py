# next version of sklearn has a similar methode implemended

#creates and returns a truth table with dimension 2n (n number of features) and prints it
#each row contains the acutal label
#each column contains the predicted label

'''
y_test_enc is the integer encoded version of label data
X_test_vect is a vector containing the test data (integer)
prediction is the predicted (by a classifier) labels of the test data
threshold is an optinal float between 0 and 1. It determines the threashold for a predicted value (label) to be true
returns a matrix containing information about the predictions (sum of labels predictet to be label x for each label y). Each row covers the real (preset) label, each column the predicted label. For example the value in [2][3], is the sum of data with preset value 2 and prediction 3.
TODO: refine text
This function also prints the matrix.
'''
def create_confusion_matrix(y_test_enc, X_test_vect, prediction, threshold = 0.5):
    matrix_range = y_test_enc.shape[1]           # number of features

    #create empty matrix
    matrix = [[0 for x in range(matrix_range +2)] for y in range(matrix_range)] 

    #enter values in matrix
    for i in range(X_test_vect.shape[0]):
        for label in range(matrix_range):
            if y_test_enc[i][label] == 1.0:
                matrix[label][matrix_range +1] +=1
                
                value_missing= True
                for label_pred in range(matrix_range):
                    if prediction[i][label_pred] >= threshold:
                        matrix[label][label_pred] += 1
                        value_missing = False
                if value_missing:
#                    print(matrix)
                    matrix[label][matrix_range] += 1
            



    #print pretty
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print ('\n'.join(table))
    
    return matrix

#------------------------------------------------------------------------------#

# use dataFrame instead
# get label names in list
# create dataFrame from label name list

'''
matrix a representation of the labels and their predictions
returns a matrix representing the percentage distribution for each row (what percentage of label x was predicted to be x, y, z,...)
'''
def create_percent_matrix(matrix):
    
    size = len(matrix)
    
    #create empty matrix
    percent_matrix = [[0 for x in range(size +1)] for y in range(size)] 
    
    for i in range(size):
        sum = matrix[i][size+1]
        for j in range(size+1):
            percent_matrix[i][j] = round(matrix[i][j] / sum,4)
            
    
    #print pretty
    s = [[str(e) for e in row] for row in percent_matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print ('\n'.join(table))
    
    return percent_matrix

################################################################################################################################################
#################################################### Helper functions ##########################################################################
'''
matrix is an arraylist containing integers
i is an integer that indicates which row should be summed up
returns the sum of row i for given matrix
'''
def sum_row(matrix, i):
    result = 0
    for j in range(len(matrix) +1):            # +1 needed because there is an extra column (no label predicted)
        result += matrix[i][j]
    return result

'''
matrix is an arraylist containing integers
i is an integer that indicates which column should be summed up
returns the sum of column i for given matrix
'''
def sum_col(matrix, i):
    result = 0
    for j in range(len(matrix)):
        result += matrix[j][i]
    return result

#------------------------------------------------------------------------------#

'''
matrix is an arraylist containing integers
row is an integer that represents label
returns the rate of entries falsely predicted to be given label(row) 
'''
def calculate_truefalse(matrix, row):
    result = 0
    sum= sum_row(matrix, row)
    pred_false= sum-matrix[row][row]
    result= pred_false/sum
    return result

'''
matrix is an arraylist containing integers
col is an integer that represents label
returns the rate of entries failed to be predicted given label(row) 
'''
def calculate_falsetrue(matrix, col):
    result = 0
    sum= sum_col(matrix, col)
    pred_false= sum-matrix[col][col]
    result= pred_false/sum
    return result

'''
matrix is an arraylist containing integers
i is an integer that represents label
returns the rate of entries preset as label i without a prediction
'''
def calculate_not_predicted(matrix, i):
    result = 0
    sum= sum_row(matrix, i)
    missing= matrix[i][len(matrix)]
    result= missing/sum
    return result

################################################################################################################################################
#################################################### Print functions  ##########################################################################
'''
y_train is label data set of the trainings data                             # TODO: only used to get all labels. Better name needed
matrix is a matrix representing the pridictions for the trained data
prints the 
'''
def print_truth_values(y_train, matrix, convert_label_map):
    
#    report_list =[]

    for i in range(y_train.shape[1]):
        rate_truefalse = calculate_truefalse(matrix, i)
        rate_falsetrue = calculate_falsetrue(matrix, i)
        not_predicted = calculate_not_predicted(matrix, i)

        if convert_label_map:
            inverted = convert_label_map['label_encoder'].inverse_transform([i])
        
        print(inverted + ' has a truefalse rate of %1.5f' %rate_truefalse)
        print(inverted + ' has a falsetrue rate of %1.5f' %rate_falsetrue)
        print(inverted + ' no label was set %1.5f' %not_predicted)
        print(inverted + ' has {0} entries'.format(sum_row(matrix, i)))
        print('=====================================================')

        

################################################################################################################################################
#################################################### Print functions  ##########################################################################

from sklearn.metrics import f1_score

def print_f1(y_true, y_pred):
    '''
    y_true acutal values of the test prediction data
    y_pred predicted values by the classifier
    prints the F1 scores for a binary classification prediction
    '''
    
    print("F1 Score (macro, micro, weighted): ")
    print(f1_score(y_true, y_pred, average='macro'))
    f1 = f1_score(y_true, y_pred, average='micro')
    print(f1)  
    print(f1_score(y_true, y_pred, average='weighted'))  
    return f1

#------------------------------------------------------------------------------#

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

def print_AUC(X_test, prob, label):
    '''
    X_test test data
    prob is the probability for each test set
    label of the prediction
    prints the AUC for a binary classification prediction
    '''
    
    # calculate AUC
    auc = roc_auc_score(X_test, prob)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(X_test, prob)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    # show the plot
    pyplot.title(label)
    pyplot.show()
    

#------------------------------------------------------------------------------#    
    
from sklearn.utils.fixes import signature
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

def print_prec_rec(y_test, pred):
    '''
    y_test are the labels for the test data
    pred is the prediction
    prints a precision recall diagram for a binary classification prediction
    '''
    precision, recall, _ = precision_recall_curve(y_test, pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(pyplot.fill_between).parameters
                   else {})
    pyplot.step(recall, precision, color='b', alpha=0.2,
             where='post')
    pyplot.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    print(type(recall))
    print(recall)
    print(type(precision))
    print(precision)

    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.ylim([0.0, 1.05])
    pyplot.xlim([0.0, 1.0])
    pyplot.show()
    
################################################################################################################################################
#################################################### Utility functions  ##########################################################################   
    
from numpy import array

def convert_probability_list(list):
    '''
    takes a list of predicted probabilities (contains a list for each prediction with % for true and false)
    returns a list only containing the probability of a positiv label
    '''
    dummyList = []
    for l in list:
        dummyList.append(l[-1])
    
    return array(dummyList)

#------------------------------------------------------------------------------#

def convert_probability_trueFalse(array, threshold):
    '''
    array containing the probabily predictions
    threshold at which a label shall be labeled as true
    returns an array containing only 0 and 1
    '''
    resultarray = array.copy()
    for i in range(0, len(array)):
        if resultarray[i] >= threshold:
            resultarray[i] = 1
        else:
            resultarray[i] = 0
    return resultarray


################################################################################################################################################
#################################################### Grid Search  ##########################################################################  
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

class ClassifierConfiguration:              
    '''
    created by do_gridsearch. Contains configuration information about a classifier
    '''
    
    def __init__(self, label, classifier, c_value, tfidf_value, acc):
        self.label = label
        if isinstance(classifier, LogisticRegression):
            self.logReg = LogisticRegression(C=c_value)
            self.logRegTfidf = tfidf_value
            self.logRegAcc = acc
        if isinstance(classifier, LinearSVC):
            self.svc = LinearSVC(C = c_value)
            self.svcTfidf = tfidf_value
            self.svcAcc = acc

#------------------------------------------------------------------------------#

class ClassifierContainer:
    def __init__(self, label, classifier, vectorizer, svd):
        self.label = label
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.svd = svd
        self.thrashold = 0.5    
    
#------------------------------------------------------------------------------#

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

from preparing_data import *
from sklearn.preprocessing import LabelEncoder

def do_gridsearch(df_map, classifier, tfidf_values = (0.15, 0.25), c_values = [0.1, 0.25, 0.5, 1, 5, 10]):
    '''
    df_map is a map<label, df> containing the label and the associated dataframe
    classifier which shall be used
    tfidf_values which values tfidf should consider 
    c_values which the classifier (logistic regression, svd) should consider
    '''
    classifierContainerList = {}
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', min_df=5)),
        ('classifier', classifier),
    ])

    parameters_pipeline = {
        'tfidf__max_df': tfidf_values,
        'classifier__C': c_values,
    }

    grid_search = GridSearchCV(pipeline, parameters_pipeline, n_jobs=-1, cv=3)

    for key, df in df_map.items():
        print("Dataset: "+key)

        #splitting data
        split_map_all = split_dataset(df)
        X_train_all = split_map_all['X_train']
        y_train_all = split_map_all['y_train']
        X_test_all = split_map_all['X_test']
        y_test_all = split_map_all['y_test']

        #encoding label
        integer_encoded_train = y_train_all.replace("NONE", 0).replace(key, 1)
        integer_encoded_test = y_test_all.replace("NONE", 0).replace(key, 1)

        grid_search.fit(X_train_all, integer_encoded_train)
        y_prediction = grid_search.predict(X_test_all)

        print(grid_search.best_params_)
        classifierContainerList[key] = ClassifierConfiguration(key, classifier, grid_search.best_params_['classifier__C'], grid_search.best_params_['tfidf__max_df'],
                                                               accuracy_score(integer_encoded_test,y_prediction))
        print(grid_search.best_score_)
        pipeline_best = grid_search.best_estimator_
        accuracy = pipeline_best.score(X_test_all, integer_encoded_test)
        print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
        
    return classifierContainerList
        
################################################################################################################################################
#################################################### Label Information  ##########################################################################  
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def print_label_distribution(df, classifier = LinearSVC(C=0.7, random_state=42)):
    '''
    df for multi label classification
    classifier which should be used for the evaluation
    prints truth table (next sklearn version has inbuild function for this)
    prints falsetrue truefalse ratio per label
    '''
    
    # splitting data
    split_map = split_dataset(df)
    X_train = split_map['X_train']
    y_train = split_map['y_train']
    X_test = split_map['X_test']
    y_test = split_map['y_test']

    # neccessary for multilabeling
    convert_label_map = convert_labels_for_multilabeling(y_train, y_test)
    y_train = convert_label_map['y_train_encoded']
    y_test = convert_label_map['y_test_encoded']
    
    tfidf_vectorizer =  TfidfVectorizer(stop_words='english', max_df=0.15)
    X_train_vect = tfidf_vectorizer.fit_transform(X_train)
    X_test_vect = tfidf_vectorizer.transform(X_test)

    # classifier may take one
    cfl= OneVsRestClassifier(classifier)
    cfl.fit(X_train_vect, y_train)

    from sklearn.metrics import classification_report
    prediction = cfl.predict(X_test_vect)
    #predictions_binary = (prediction > 0.4).as_type(np.float64)
    report = classification_report(y_test, prediction)
    print(report)

    # in the new sklearn version there will be an inbuild function for truthtables for multilabel classification
    matrix = create_confusion_matrix(y_test, X_test_vect, prediction, 0.5)
    print_truth_values(y_train, matrix, convert_label_map)

################################################################################################################################################
#################################################### Determine Threshold  ##########################################################################  