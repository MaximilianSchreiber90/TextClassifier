from sklearn.datasets import fetch_20newsgroups 
import pandas as pd

def get_20news():
    '''
    fetches the 20 newsgroup dataset from sklearn.datasets. The data will be converted to a dataframe with the two columns text, category.
    In the dataset there is 20 different labels across seven main label classes. For this project we only use nine out of the 20 labels and
    map them to three labels: SPORTS, SCIENCE, POLITICS
    '''
    
    #relevant datasets ordered by category
    sports_cat = ['rec.sport.baseball',
                  'rec.sport.hockey']
    science_cat = ['sci.crypt',
                   'sci.electronics',
                   'sci.med',
                   'sci.space']
    politics_cat = ['talk.politics.guns',
                    'talk.politics.mideast',
                    'talk.politics.misc']
    
    #fetiching the data
    sports_data_20news = fetch_20newsgroups(categories=sports_cat, random_state=42)
    sports_data_20news = pd.DataFrame(data={'text': sports_data_20news.data})
    sports_data_20news['category'] = 'SPORTS'
    science_data_20_news = fetch_20newsgroups(categories=science_cat, random_state=42)
    science_data_20_news = pd.DataFrame(data={'text': science_data_20_news.data})
    science_data_20_news['category'] = 'SCIENCE'
    politics_data_20news = fetch_20newsgroups(categories=politics_cat, random_state=42)
    politics_data_20news = pd.DataFrame(data={'text': politics_data_20news.data})
    politics_data_20news['category'] = 'POLITICS'

    # picking only relevant columns
    selected_columns = ['text', 'category']
    df = sports_data_20news.copy()
    df = df.append(science_data_20_news.copy())
    df = df.append(politics_data_20news.copy())
    
    return df

################################################################################################################################################
    
def huff_prepared():
    '''
    returns all data from huffpost (only relevant columns)
    '''
    return only_relevant_columns(combine_huff_categories(get_huffpost()))

#------------------------------------------------------------------------------#

import os

def get_huffpost():
    '''
    reads the News_Category_Dataset.json and returns its data as dataframe. Also combines the headline text with the short summary in the new column text
    '''
    
    #getting proper path
    script_dir = os.getcwd()
    file_path = os.path.join(script_dir, 'Data\\News_Category_Dataset_new.json')
    
    # loading data
    df = pd.read_json(file_path, lines=True)
    
    # adding headline and description together as text
    df['text'] = df.headline + " " + df.short_description
    
    return df
    
#------------------------------------------------------------------------------#

def combine_huff_categories(df):
    '''
    df is the dataframe containing the data from get_huffpost()
    This function maps and removes labels according to the documentation and returns the results
    '''
    # ENTERTAINMENT
    df.category = df.category.map(lambda x: "ENTERTAINMENT" if x == "ARTS" else x)
    df.category = df.category.map(lambda x: "ENTERTAINMENT" if x == "ARTS & CULTURE" else x)
    df.category = df.category.map(lambda x: "ENTERTAINMENT" if x == "COMEDY" else x)
    df.category = df.category.map(lambda x: "ENTERTAINMENT" if x == "RELIGION" else x)
    df.category = df.category.map(lambda x: "ENTERTAINMENT" if x == "STYLE" else x)
    df.category = df.category.map(lambda x: "ENTERTAINMENT" if x == "TASTE" else x)
    df.category = df.category.map(lambda x: "ENTERTAINMENT" if x == "TRAVEL" else x)
    df.category = df.category.map(lambda x: "ENTERTAINMENT" if x == "ENTERTAINMENT" else x)

    # HEALTH
    df.category = df.category.map(lambda x: "HEALTH" if x == "GREEN" else x)
    df.category = df.category.map(lambda x: "HEALTH" if x == "HEALTHY LIVING" else x)
    #SCIENCE
    df.category = df.category.map(lambda x: "SCIENCE" if x == "TECH" else x)
    # BUSINESS

    # SPORTS

    # POLITICS
    
    # FOOD
    df.category = df.category.map(lambda x: "FOOD" if x == 'FOOD & DRINK' else x)
    
    # MONEY
    
    # Delete entries   
    remove_category_list = ['BLACK VOICES', 'WEIRD NEWS', 'QUEER VOICES', 'LATINO VOICES', "FIFTY", "GOOD NEWS", "PARENTS", "MEDIA", "CRIME", "WOMEN", "IMPACT", "THE WORLDPOST", "WORLD NEWS", "COLLEGE",
        'EDUCATION', 'WORLDPOST', 'WELLNESS','PARENTING','HOME & LIVING','STYLE & BEAUTY','DIVORCE','WEDDINGS','ENVIRONMENT','CULTURE & ARTS']
    for label in remove_category_list:
        df = df[df.category != label]
    
    return df

#------------------------------------------------------------------------------#

def only_relevant_columns(df):
    '''
    df is the dataframe containing the data from get_huffpost()
    returns a copy of the df containing only the two relevant columns text and category
    '''
    selected_columns = ['text', 'category']
    return df[selected_columns].copy()   
    
##############Combined Dataframes##################

def get_all_data_prepared(min_len = 7):
    '''
    min_len is the minimum word count for a dataset to not be deleted
    fetches all data, deletes and combines labels and returns them as a df. Also removes short datasets.
    '''
    huff_data = only_relevant_columns(combine_huff_categories(get_huffpost()))
    df_20news = get_20news()
    
    df = huff_data.append(df_20news)
    
    df = add_word_vector_column(df)
    df = delete_short_datasets(df, min_len)
    
    return df

#------------------------------------------------------------------------------#

def get_df_map(ratio = 0.3):
    '''
    ratio of positive label in each dataframe
    fetches all data and returns a map<label, df> containing a binary dataset for each label
    '''
    if ratio < 0 or ratio > 1:
        ratio = 0.3
    
    df = get_all_data_prepared()
    dataset_map = {}
    
    categories = df['category'].unique()
    for cat in categories:
        dataset_map[cat] = get_test_data_for_label(df, cat, ratio)
        
    return dataset_map

##############Cleaning the Data##################

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer


def letters_only(astr):
    '''
    astr is a Sting of characters.
    returns true if all characters are letters otherwise false.
    This function is used by clean_text() to remove numers from the text
    '''
    for c in astr:
        if not c.isalpha():
            return False
    return True

#------------------------------------------------------------------------------#

# not used
def no_noune(astr):
    '''
    astr is a Sting of characters.
    returns true if all characters are letters otherwise false.
    This function is used by clean_text() to remove numers from the text
    '''
    for c in astr:
        if not c.isalpha():
            return False
    return True

#------------------------------------------------------------------------------#

def clean_text(docs):
    '''
    docs is a text string
    returns a cleaned version of input string, by using a lemmatizer and removing all non letters
    '''
    lemmatizer = WordNetLemmatizer()
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                        for word in doc.split()
                                        if letters_only(word)]))
    return cleaned_docs
#------------------------------------------------------------------------------#

def add_word_vector_column(df):
    '''
    df is a dataframe
    returns input dataframe, but adds a new column named words. This column contains the version of the text of the dataframe
    '''
    df['words'] = clean_text(df.text)
    return df

#------------------------------------------------------------------------------#

def delete_short_datasets(df, min_len=7):
    '''
    df is a dataframe
    removes the short data (< 7 words) from input data and returns the result
    '''
    df['word_length'] = df.words.apply(lambda i: len(i))
    return df[df.word_length > min_len]

##############Utility##################
import math

def get_test_data_for_label(df, labelname, split_ratio):
    '''
    Takes a dataFrame and prepares a copy of it for single label classification
    df: dataFrame which should be prepared
    labelname: name of the label which should be classified
    split_size: ratio of label/data entries
    returns a randomly generated dataframe contining selected label including nose data from all other labels
    '''
    #create new DF containing only selected label
    result_df = df[df.category == labelname].copy()
    
    label_size = result_df.category.count()
    
    #if(result_df.size == 0):
        #error message
        
    random_df = df[df.category != labelname].copy()
    random_df['category'] = 'NONE'             # check if exist
    random_size = random_df.category.count()
        
    random_size_needed = int(math.ceil(label_size / split_ratio))
    
    if(random_size_needed < random_size):
        result_df= result_df.append(random_df.sample(n = random_size_needed))
    else:
        print("There were not enough data to create a split with split_ratio {:}. Only selected {:} instead of {:} random data".format(split_ratio, random_size, random_size_needed))
        result_df= result_df.append(random_df.sample(n = random_size))

    return result_df

#------------------------------------------------------------------------------#   
 
def get_size_for_all_labels(df):
    '''
    df is dataframe
    categories is a list of category strings which should be counted. If no list is given, the size of all labels will be returned
    returns the size of each label for given dataset and category list
    '''   
    categories = df['category'].unique()
    for cat in categories:
        print('{} contains {} entries'.format(cat, len(df[df.category==cat])))
        
#------------------------------------------------------------------------------#      

import nltk
import sklearn
'''
prints versions of used packages
'''
def print_versions():
    print('The nltk version is {}.'.format(nltk.__version__))
    print('The scikit-learn version is {}.'.format(sklearn.__version__))