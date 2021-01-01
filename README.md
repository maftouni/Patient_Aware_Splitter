# Patient_Aware_Image_Split
This repository splits a sample Covid/Normal classification dataset into validation and training sets in a patient aware manner. It is important not to split images of the same patient between the sets (to avoid overfitting); therefore, we have used meta-data file to group the images based on Patient-ID first. 

It is also important to have stratified split of Covid/NonCovid cases. As we only have two classes here, I have disentagled stratification from grouping by splitting each of the classes separately. 

**split_to_folders.py** splits images into 4 folders inside splitted: train/Covid, train/NonCovid, test/Covid, test/NonCovid
