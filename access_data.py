from sklearn.datasets import fetch_20newsgroups 
import pandas as pd

'''
fetches the 20 newsgroup dataset from sklearn.datasets. The data will be converted to a dataframe with the two columns text, category.
In the dataset there is 20 different labels across seven main label classes. For this project we only use nine out of the 20 labels and
map them to three labels: SPORTS, SCIENCE, POLITICS
'''
def get_20news():
    
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
    
#    print('size sports 20 news: {}'.format(len(sports_data_20news.text)))
#    print('size science 20 news: {}'.format(len(science_data_20_news.text)))
#    print('size politics 20 news: {}'.format(len(politics_data_20news.text)))

    # picking only relevant columns
    selected_columns = ['text', 'category']
    df = sports_data_20news.copy()
    df = df.append(science_data_20_news.copy())
    df = df.append(politics_data_20news.copy())
    
    return df

################################################################################################################################################
import os

'''
reads the News_Category_Dataset.json and returns its data as dataframe. Also combines the headline text with the short summary in the new column text
'''
def get_huffpost():
    #getting a proper path
    script_dir = os.getcwd()
    file_path = os.path.join(script_dir, 'Data\\News_Category_Dataset_new.json')
    
    print(file_path)
    
    # loading data
    df = pd.read_json(file_path, lines=True)
    
    # adding headline and description together as text
    df['text'] = df.headline + " " + df.short_description
    
    return df
    
#------------------------------------------------------------------------------#

'''
df is the dataframe containing the data from get_huffpost()
This function mapps the labels according to the documentation and returns the results
'''
def combine_huff_categories(df):
    df.category = df.category.map(lambda x: "LIFESTYLE" if x == "ARTS" else x)
    df.category = df.category.map(lambda x: "LIFESTYLE" if x == "ARTS & CULTURE" else x)
    df.category = df.category.map(lambda x: "LIFESTYLE" if x == "COMEDY" else x)
    df.category = df.category.map(lambda x: "LIFESTYLE" if x == "RELIGION" else x)
    df.category = df.category.map(lambda x: "LIFESTYLE" if x == "STYLE" else x)
    df.category = df.category.map(lambda x: "LIFESTYLE" if x == "TASTE" else x)
    df.category = df.category.map(lambda x: "LIFESTYLE" if x == "TRAVEL" else x)
    df.category = df.category.map(lambda x: "LIFESTYLE" if x == "ENTERTAINMENT" else x)

    # HEALTH
    df.category = df.category.map(lambda x: "HEALTH" if x == "GREEN" else x)
    df.category = df.category.map(lambda x: "HEALTH" if x == "HEALTHY LIVING" else x)
    #SCIENCE
    df.category = df.category.map(lambda x: "SCIENCE" if x == "TECH" else x)
    # BUSINESS

    # SPORTS

    # POLITICS

    # EDUCATION
    df.category = df.category.map(lambda x: "EDUCATION" if x == "COLLEGE" else x)

    # IMPACT

    # WORLDPOST
    df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
    df.category = df.category.map(lambda x: "WORLDPOST" if x == "WORLD NEWS" else x)

    # RANDOM
    df.category = df.category.map(lambda x: "RANDOM" if x == "FIFTY" else x)
    df.category = df.category.map(lambda x: "RANDOM" if x == "GOOD NEWS" else x)
    df.category = df.category.map(lambda x: "RANDOM" if x == "PARENTS" else x)
    df.category = df.category.map(lambda x: "RANDOM" if x == "MEDIA" else x)
    df.category = df.category.map(lambda x: "RANDOM" if x == "CRIME" else x)
    df.category = df.category.map(lambda x: "RANDOM" if x == "WOMEN" else x)

    # Delete entries
    df = df[df.category != 'BLACK VOICES']    #POLITICS
    df = df[df.category != 'WEIRD NEWS']
    df = df[df.category != 'QUEER VOICES']
    df = df[df.category != 'LATINO VOICES']
    
    return df


#------------------------------------------------------------------------------#

def dropping_new_category(df):
    new_category_list = ['WELLNESS','PARENTING','HOME & LIVING','STYLE & BEAUTY','DIVORCE','WEDDINGS','FOOD & DRINK','MONEY','ENVIRONMENT','CULTURE & ARTS']
    
    for label in new_category_list:
        df = df[df.category != label]
    
    return df


#------------------------------------------------------------------------------#
'''
df is the dataframe containing the data from get_huffpost()
returns a copy of the df containing only the two relevant columns text and category
'''
def only_relevant_columns(df):
    selected_columns = ['text', 'category']
    return df[selected_columns].copy()

'''
loads all huffpost data, combines categories and deletes irrelevant columns
'''
def prepared_huffpost():
    return only_relevant_columns(combine_huff_categories(get_huffpost()))
   
    
##############Combining Dataframes##################

##############Cleaning the Data##################

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

'''
astr is a Sting of characters.
returns true if all characters are letters otherwise false.
This function is used by clean_text() to remove numers from the text
'''
def letters_only(astr):
    for c in astr:
        if not c.isalpha():
            return False
    return True

#------------------------------------------------------------------------------#

'''
astr is a Sting of characters.
returns true if all characters are letters otherwise false.
This function is used by clean_text() to remove numers from the text
'''
def no_noune(astr):
    for c in astr:
        if not c.isalpha():
            return False
    return True

#------------------------------------------------------------------------------#

'''
docs is a text string
returns a cleaned version of input string, by using a lemmatizer and removing all letters
'''
def clean_text(docs):
    lemmatizer = WordNetLemmatizer()
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                        for word in doc.split()
                                        if letters_only(word)]))
    return cleaned_docs
#------------------------------------------------------------------------------#
'''
df is a dataframe
returns input dataframe, but adds a new column named words. This column contains the version of the text of the dataframe
'''
def add_word_vector_column(df):
    df['words'] = clean_text(df.text)
    return df

#------------------------------------------------------------------------------#
'''
df is a dataframe
removes the short data (<= 6 words) from input data and returns the result
'''
def delete_short_datasets(df):
    df['word_length'] = df.words.apply(lambda i: len(i))
    return df[df.word_length >= 6]

##############Utility##################
import math

'''
Takes a dataFrame and prepares a copy of it for single label classification
df: dataFrame which should be prepared
labelname: name of the label which should be classified
split_size: ratio of label/data entries
returns a randomly generated dataframe contining selected label including nose data from all other labels
'''
def get_test_data_for_label(df, labelname, split_ratio):
   
    #create new DF containing only selected label
    result_df = df[df.category == labelname]
    
    label_size = result_df.category.count()
    
    #if(result_df.size == 0):
        #error message
        
    random_df = df[df.category != labelname]
    random_df['category'] = 'NONE'
    random_size = random_df.category.count()
        
    random_size_needed = int(math.ceil(label_size / split_ratio))
    
    if(random_size_needed < random_size):
        result_df= result_df.append(random_df.sample(n = random_size_needed))
    else:
        print("There were not enough data to create a split with split_ratio {:}. Only selected {:} instead of {:} random data".format(split_ratio, random_size, random_size_needed))
        result_df= result_df.append(random_df.sample(n = random_size))

    return result_df

#------------------------------------------------------------------------------#   
'''
df is dataframe
categories is a list of category strings which should be counted. If no list is given, the size of all labels will be returned
returns the size of each label for given dataset and category list
'''    
def get_size_for_all_labels(df, categories=['SCIENCE', 'SPORTS', 'POLITICS', 'LIFESTYLE', 'HEALTH', 'EDUCATION', 'WORLDPOST', 'IMPACT', 'BUSINESS', 'RANDOM']):   
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