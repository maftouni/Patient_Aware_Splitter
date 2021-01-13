from sklearn.model_selection import train_test_split
from shutil import copyfile
import pandas as pd
import glob



def Stratified_Grouped_train_test_split(df, group, stratify_by,test_size=0.2):
    # I want grouping to be done strictly to ensure there is no overlap of groups whilst stratification 
    # can be done approximately i.e. as well as is possible.
    #This function assumes that all instances of one group have the same stratification category,
    # In other words, all the images coming from the same Patient ID are either Covid or NonCovid
    
    groups = df[group].drop_duplicates()
    stratify = df.drop_duplicates(group)[stratify_by].to_numpy()
    groups_train, groups_test = train_test_split(groups, stratify=stratify, test_size=test_size)

    train = df.loc[lambda d: d[group].isin(groups_train)]
    test = df.loc[lambda d: d[group].isin(groups_test)]

    return train, test


def splitter(df, group, stratify_by):
    # The function splits the given dataframe (df) after grouping by the group column (which is 'Patient ID' in our case)
    # and Stratifying by stratify_by column ('COVID-19 Infection' in this case)

    train, test = Stratified_Grouped_train_test_split(df, group, stratify_by)
    
    train_filenames = list(train['File name'])
    test_filenames = list(test['File name'])
    
    count = 0


    image_list = glob.glob('sample_data'+'/*.png')
    
    for j in range(len(image_list)): 
                ## modify to .split('\\') according to your system path  
                image = image_list[j].split('/')[-1]  
                
                if image in train_filenames: 
                    if train.loc[df['File name'] == image, 'COVID-19 Infection'].values[0] == 'Positive':
                       count +=1
                       copyfile(image_list[j], 'splitted/train/Covid/'+image_list[j].split('/')[-1])
                    else:
                       count +=1
                       copyfile(image_list[j], 'splitted/train/NonCovid/'+image_list[j].split('/')[-1])
                elif image in test_filenames:
                    if test.loc[df['File name'] == image, 'COVID-19 Infection'].values[0] == 'Positive':  
                        count +=1
                        copyfile(image_list[j], 'splitted/test/Covid/'+image_list[j].split('/')[-1])
                    else:
                       count +=1    
                       copyfile(image_list[j], 'splitted/test/NonCovid/'+image_list[j].split('/')[-1])
                
                    
    print('Total number of copied images: ',count)
    print(' ')
