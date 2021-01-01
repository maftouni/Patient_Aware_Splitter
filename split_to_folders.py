import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import glob
import os
from shutil import copyfile

def splitter(df,class_name,grouping_column):
    # The function splits the given dataframe (df) after grouping by the grouping_column (which is 'Patient ID' in our case)
    # The classes are separated beforehand to disentagle class stratification and patient grouping problems
    
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state = 1).split(df, groups=df[grouping_column]))
    print('Number of '+ class_name +' train images: ',len(train_inds))
    print('Number of '+ class_name +' test images: ',len(test_inds))
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    
    train_filenames = list(train['File name'])
    train_diagnosis =list(train['COVID-19 Infection'])
    test_filenames = list(test['File name'])
    test_diagnosis =list(test['COVID-19 Infection'])
    
    #print(train_diagnosis)
    count = 0


    image_list = glob.glob('sample_data'+'/*.png')
    
    for j in range(len(image_list)):          
                image = image_list[j].split('/')[-1]                
                if image in train_filenames: 
                       count +=1
                       copyfile(image_list[j], 'splitted/train/'+class_name+'/'+image_list[j].split('/')[-1])
                elif image in test_filenames:
                        count +=1
                        copyfile(image_list[j], 'splitted/test/'+class_name+'/'+image_list[j].split('/')[-1])
                
                    
    print('Total number of copied '+class_name+' images: ',count)
    print(' ')
    
 


 
    
    
df = pd.read_csv('sample_meta_data.csv')
df['Patient ID'] = df['Patient ID'].astype(str)

covid = df[df['COVID-19 Infection'] == 'Positive']
noncovid = df[df['COVID-19 Infection'] == 'Negative']

splitter(covid,'Covid','Patient ID')
splitter(noncovid,'NonCovid','Patient ID')
