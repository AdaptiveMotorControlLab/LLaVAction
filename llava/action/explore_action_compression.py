import pandas as pd
import sys
import os

def compare_noun_v1_v2():
    noun_v1_csv_path = '../epic-kitchens-100-annotations/EPIC_100_noun_classes.csv'
    noun_v2_csv_path = '../epic-kitchens-100-annotations/EPIC_100_noun_classes_v2.csv'

    df1 = pd.read_csv(noun_v1_csv_path)
    df2 = pd.read_csv(noun_v2_csv_path)

    print (df1)
    print (df2)

    diff_in_keys = df1['key'] == df2['key'][:300]
    print ('diff_in_keys', diff_in_keys)
    print ('diff_in_keys sum', diff_in_keys.sum())

    diff_in_instances = df1['instances']==df2['instances'][:300]
    # sum number of true in diff_in_instances
    print ('diff_in_instances', diff_in_instances)
    print ('diff_in_instances sum', diff_in_instances.sum())

def calculate_compression_rate():
    noun_v1_csv_path = '../epic-kitchens-100-annotations/EPIC_100_noun_classes.csv'
    noun_v1 = pd.read_csv(noun_v1_csv_path)
    print(noun_v1.iloc[0]['instances'])
    num_compressed_noun = len(noun_v1['key'])
    print ('num_compressed_noun v1', num_compressed_noun)
    # instances are list of instances of the noun. I want to get the total number of instances
    instances = noun_v1['instances']
    # apply lambda function to get length of instances
    instance_len = instances.apply(lambda x: len(eval(x)))
    print ('instance_len noun v1', instance_len)
    print ('sum of instance_len noun v1', instance_len.sum())
    print ('noun v1 compression rate', instance_len.sum()/num_compressed_noun)


    noun_v2_csv_path = '../epic-kitchens-100-annotations/EPIC_100_noun_classes_v2.csv'
    noun_v2 = pd.read_csv(noun_v2_csv_path)
    print(noun_v1.iloc[0]['instances'])
    num_compressed_noun = len(noun_v2['key'])
    print ('num_compressed_noun v2', num_compressed_noun)
    # instances are list of instances of the noun. I want to get the total number of instances
    instances = noun_v2['instances']
    print ('instances')
    print (instances)
    # apply lambda function to get length of instances
    instance_len = instances.apply(lambda x: len(eval(x)))
    print ('instance_len noun v2', instance_len)
    print ('sum of instance_len noun v2', instance_len.sum())
    print ('noun v2 compression rate', instance_len.sum()/num_compressed_noun)    

    verb_csv_pah = '../epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
    verb = pd.read_csv(verb_csv_pah)
    num_compressed_verbs = len(verb['key'])
    print ('num_compressed_verbs', num_compressed_verbs)
    instances = verb['instances']
    instance_len = instances.apply(lambda x: len(eval(x)))
    print ('instance_len', instance_len)
    print ('sum of instance_len', instance_len.sum())
    print ('verb compression rate', instance_len.sum()/num_compressed_verbs)     

def explore_verb():
    verb_csv_path = '../epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
    verb = pd.read_csv(verb_csv_path)
    verb_list = verb['key']
    print ('list of verb centroids:')
    print (verb_list.to_list())

def explore_verb_and_avion():
    """
    Check whether avion uses verb centroid or narration verb
    """
    pass

def whether_narration_match_centroids():
    """
    Whether the verb and noun in narration match the centroid keys in verb and noun csv.
    """
    verb_csv_path = '../epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
    verbs = pd.read_csv(verb_csv_path)
    verb_centroid_list = verbs['key']

    noun_csv_path = '../epic-kitchens-100-annotations/EPIC_100_noun_classes_v2.csv'
    nouns = pd.read_csv(noun_csv_path)
    noun_centroid_list = nouns['key']

    validation_csv_path = '../epic-kitchens-100-annotations/EPIC_100_validation.csv'
    validation = pd.read_csv(validation_csv_path)
    print ('narration')
    narration = validation['narration']
    print (narration)
    def assert_check(e):
        print (e)
        if len(e.split(' ')) > 2:
            # perhaps wrong
            verb = e.split(' ')[0]
            noun = e.split(' ')[1]
        else:
            verb, noun = e.split(' ')
        if verb not in verb_centroid_list:
            print ('verb not in verb_centroid_list', verb)
        else:
            print ('verb in verb_centroid_list', verb)
        if noun not in noun_centroid_list:
            print ('noun not in noun_centroid_list', noun)
        else:
            print ('noun in noun_centroid_list', noun)

    narration.apply(assert_check)


def generate_explore():
    # verb csv path has the clustering of verbs
    verb_csv_path = '../epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
    verb_classes = pd.read_csv(verb_csv_path)
    print (verb_classes)
    # count number of unique key in verb_classes
    print ('verb_class count ', verb_classes['key'].value_counts())
    validation_csv_path = '../epic-kitchens-100-annotations/EPIC_100_validation.csv'
    validation = pd.read_csv(validation_csv_path)
    # print the column verb and verb_class together
    print (validation.columns)

    df = validation[['narration', 'verb', 'verb_class', 'noun', 'noun_class']]
    print (df)
    print ('verb_class count', validation['verb_class'].value_counts())
    print ('noun_class count', validation['noun_class'].value_counts())
    print ('verb count', validation['verb'].value_counts())
    print ('noun count', validation['noun'].value_counts())


    # # check whether sum of verb count and verb_class count be equal?
    # print ('verb_class count sum', validation['verb_class'].value_counts().sum())
    # print ('verb count sum', validation['verb'].value_counts().sum())
    # # check whetehr sum of noun count and noun_class count be equal?
    # print ('noun_class count sum', validation['noun_class'].value_counts().sum())
    # print ('noun count sum', validation['noun'].value_counts().sum())


    # find rows in df where the noun_class is euqal to 2 
    samples = df[df['noun_class'] == 1]

    print(samples['noun'].value_counts())


#compare_noun_v1_v2()
#generate_explore()
#calculate_compression_rate()
whether_narration_match_centroids()