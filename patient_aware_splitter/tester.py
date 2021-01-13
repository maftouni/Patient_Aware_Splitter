df = pd.read_csv('sample_meta_data.csv')
df['Patient ID'] = df['Patient ID'].astype(str)


#folders
splitter(df,'Patient ID','COVID-19 Infection')

# dictionary
splitted_dictionary = splitter(df,'Patient ID','COVID-19 Infection')



# data loader

df = pd.read_csv('sample_meta_data.csv')
df['Patient ID'] = df['Patient ID'].astype(str)


## Splitted into a dictionary
splitted_dictionary = splitter(df,'Patient ID','COVID-19 Infection')

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
