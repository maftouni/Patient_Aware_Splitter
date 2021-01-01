
from sklearn.model_selection import GroupShuffleSplit
import glob
import os
import cv2
import pickle
import pandas as pd
from random import shuffle
from shutil import copyfile

def splitter(df,grouping_column,pickling = False):
    # The function splits the given dataframe (df) after grouping by the grouping_column (which is 'Patient ID' in our case)
    # The classes are separated beforehand to disentagle class stratification and patient grouping problems
       
    
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state = 1).split(df, groups=df[grouping_column]))
    print('Number of train images: ',len(train_inds))
    print('Number of test images: ',len(test_inds))
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    
    train_filenames = list(train['File name'])
    test_filenames = list(test['File name'])
    
    
    dir_images_train = []
    dir_label_train = []
    dir_images_test = []
    dir_label_test = []
    count = 0


    image_list = glob.glob('sample_data'+'/*.png')
    
    
    
    for j in range(len(image_list)):          
                image = image_list[j].split('/')[-1]                
                if image in train_filenames: 
                       count +=1
                       img = cv2.imread(image_list[j])                       
                       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                       dir_images_train.append(img)
                       dir_label_train.append(df.loc[df['File name'] == image, 'COVID-19 Infection'].values[0])
                       #copyfile(image_list[j], 'splitted/train/'+class_name+'/'+image_list[j].split('/')[-1])
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
           
    #print(diction['y_tr'])
    #print(diction['X_tr'][0].shape)
    if pickling == True:   
        with open('training.pickle', 'wb') as handle:
              pickle.dump(diction, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    return  diction
    
   
 
    
df = pd.read_csv('sample_meta_data.csv')
df['Patient ID'] = df['Patient ID'].astype(str)


splitted_dictionary = splitter(df,'Patient ID')
