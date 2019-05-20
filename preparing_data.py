'''
df is the datasate which shall be split in train and test
test_size_split is an float between 0 and 1. It determines the size of the test data
validate_set a boolean (set to false by default) which splits test set into validation and test set (50:50)
returns a map containing the following keys: 'X_train', 'y_train', 'X_test', 'y_test' 
'''
def split_dataset(df, test_size_split = 0.4, validate_set=False):

    X = df.words
    y = df.category

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size_split, random_state=0)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    if validate_set:
        sss2 = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
        sss.get_n_splits(X_test, y_test)
        for validate_index, test_index in sss2.split(X_test, y_test):
            X_validate, X_test = X.iloc[validate_index], X.iloc[test_index]
            y_validate, y_test = y.iloc[validate_index], y.iloc[test_index]
        return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test ,'y_test':y_test, 'X_validate':X_validate, 'y_validate':y_validate}
        
    return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test ,'y_test':y_test}

################################################################################################################################################

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# converts the category to a integer and the integer to an sparse(?) vector of set label
'''
y_train and y_test are the label data for the classification problem
returns a map containing the following keys: 'integer_encoded_train', 'integer_encoded_test', 'y_train_encoded', 'y_test_encoded', 'label_encoder'
This function replaces each label (Sting) by an integer. This is needed for some classifiers. The label_encoder contains the neccessary information to reverse the convertion.
'''
def convert_labels_for_multilabeling(y_train, y_test):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded_train = label_encoder.fit_transform(y_train)
    integer_encoded_test = label_encoder.transform(y_test)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_train = integer_encoded_train.reshape(len(integer_encoded_train), 1)
    y_train_encoded = onehot_encoder.fit_transform(integer_encoded_train)

    integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
    y_test_encoded = onehot_encoder.fit_transform(integer_encoded_test)
    
    return {'integer_encoded_train':integer_encoded_train, 
            'integer_encoded_test':integer_encoded_test,
            'y_train_encoded':y_train_encoded,
            'y_test_encoded':y_test_encoded,
            'label_encoder': label_encoder}