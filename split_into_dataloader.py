from sklearn.model_selection import GroupShuffleSplit
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms


def splitter(df,grouping_column):
    # The function splits the given dataframe (df) after grouping by the grouping_column (which is 'Patient ID' in our case)
    
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
    diction['Y_tr'] = dir_label_train
    diction['X_test'] = dir_images_test
    diction['Y_test'] = dir_label_test
           
    
    return  diction


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def split_into_dataloader(images,labels,nrows = 256,ncolumns = 256):
    
    X = []
    
    if np.array(images).ndim ==1:
        images = np.reshape(np.array(images),[len(images),])
    else:
        images = np.array(images)
        
    for img in list(range(0,len(images))):
        if images[img].ndim>=3:
            X.append(np.moveaxis(cv2.resize(images[img][:,:,:3], (nrows,ncolumns),interpolation=cv2.INTER_CUBIC), -1, 0))
        else:
            smimg= cv2.cvtColor(images[img],cv2.COLOR_GRAY2RGB)
            X.append(np.moveaxis(cv2.resize(smimg, (nrows,ncolumns),interpolation=cv2.INTER_CUBIC), -1, 0))
        
        if labels[img]=='Positive':
            labels[img]=1
        elif labels[img]=='Negative' :
            labels[img]=0
        else:
            continue

    x = np.array(X)
    y = np.array(labels)
    
    return x,y
    
    
    
  
    
df = pd.read_csv('sample_meta_data.csv')
df['Patient ID'] = df['Patient ID'].astype(str)


## Splitted into a dictionary
splitted_dictionary = splitter(df,'Patient ID')

## Resizing all the images and making numpy arrays from the splitted dictionary
x_train , y_train = split_into_dataloader(splitted_dictionary['X_tr'],splitted_dictionary['Y_tr'])
x_test , y_test = split_into_dataloader(splitted_dictionary['X_test'],splitted_dictionary['Y_test'])


## train/test transformations
image_transforms = { 
     'train': transforms.Compose([
         transforms.Lambda(lambda x: x/255),
        transforms.ToPILImage(), 
         transforms.Resize((230, 230)),
    transforms.RandomResizedCrop((224),scale=(0.75,1.0)),     
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    #transforms.Affine(10,shear =(0.1,0.1)),
    # random brightness and random contrast
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.45271412, 0.45271412, 0.45271412],
                             [0.33165374, 0.33165374, 0.33165374])
     ]),
     'test': transforms.Compose([
         transforms.Lambda(lambda x: x/255),
         transforms.ToPILImage(), 
       transforms.Resize((230, 230)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.45271412, 0.45271412, 0.45271412],
                             [0.33165374, 0.33165374, 0.33165374])
     ])
    }
 
## train and test Dataset and Dataloader    
train_data = MyDataset(x_train, y_train,image_transforms['train'])  
test_data = MyDataset(x_test, y_test,image_transforms['test'])


dataloaders = {
        'train' : DataLoader(train_data, shuffle=True, pin_memory=True, drop_last=False),
        'test' : DataLoader(test_data, shuffle=True, pin_memory=True, drop_last=False)  
}