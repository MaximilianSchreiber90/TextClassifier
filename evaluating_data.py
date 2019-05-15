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
    print("F1 Score (macro, micro, weighted): ")
    print(f1_score(y_true, y_pred, average='macro'))
    f1 = f1_score(y_true, y_pred, average='micro')
    print(f1)  
    print(f1_score(y_true, y_pred, average='weighted'))  
    return f1




from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

def print_AUC(X_test, prob, key):
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
    pyplot.title(key)
    pyplot.show()
    
    
from sklearn.utils.fixes import signature
from sklearn.metrics import precision_recall_curve

def print_prec_rec(y_test, pred):
    precision, recall, _ = precision_recall_curve(y_test, pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
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