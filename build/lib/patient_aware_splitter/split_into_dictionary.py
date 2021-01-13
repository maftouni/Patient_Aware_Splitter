import glob
import cv2
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


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

def splitter(df, group, stratify_by,pickling = False):
    # The function splits the given dataframe (df) after grouping by the group column (which is 'Patient ID' in our case)
    # and Stratifying by stratify_by column ('COVID-19 Infection' in this case)

    train, test = Stratified_Grouped_train_test_split(df, group, stratify_by)
    
    train_filenames = list(train['File name'])
    test_filenames = list(test['File name'])
    
    
    dir_images_train = []
    dir_label_train = []
    dir_images_test = []
    dir_label_test = []
    count = 0


    image_list = glob.glob('sample_data'+'/*.png')
    
    
    
    for j in range(len(image_list)):  
                ## modify to .split('\\') according to your system path          
                image = image_list[j].split('/')[-1]                
                if image in train_filenames: 
                       count +=1
                       img = cv2.imread(image_list[j])                       
                       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                       dir_images_train.append(img)
                       dir_label_train.append(df.loc[df['File name'] == image, 'COVID-19 Infection'].values[0])
                elif image in test_filenames:
                        count +=1
                        img = cv2.imread(image_list[j])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        dir_images_test.append(img)
                        dir_label_test.append(df.loc[df['File name'] == image, 'COVID-19 Infection'].values[0])
   
    
    print('Total number of appended images: ',count)
    print(' ')
    
    diction = {}
    diction['X_tr'] = dir_images_train
    diction['y_tr'] = dir_label_train
    diction['X_test'] = dir_images_test
    diction['y_test'] = dir_label_test
           
 
    if pickling == True:   
        with open('training.pickle', 'wb') as handle:
              pickle.dump(diction, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    return  diction
    
   
    
df = pd.read_csv('sample_meta_data.csv')
df['Patient ID'] = df['Patient ID'].astype(str)


splitted_dictionary = splitter(df,'Patient ID','COVID-19 Infection')
