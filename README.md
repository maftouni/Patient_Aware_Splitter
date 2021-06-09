# Covid_Patient_Aware_Image_Split
It is important not to split images of the same patient between the test and train sets to avoid overfitting. This repository splits a sample Covid/Normal classification dataset into test and train sets in a patient aware and stratified manner. The meta-data file is used to group the images based on Patient-ID. For example, all the images colored green belong to the same patient and should be either in the test or the train split. 

![Screenshot](patient_aware_splitter/Images_Grouped_by_Patient_ID.png)

While grouping should be done strictly to ensure there is no splitting images of a patient, stratification can be done approximately i.e. as well as possible.
This code also assumes that all images of one patient have the same stratification category (diagnosis), meaning that all the images coming from the same Patient ID are either Covid or NonCovid.

=========

[![Latest Version](https://img.shields.io/pypi/v/patient-aware-splitter)](https://pypi.org/project/patient-aware-splitter/)


Installation
------------

.. code-block:: bash

  pip install patient-aware-splitter
   
   
To split images into 4 folders (train/Covid, train/NonCovid, test/Covid, test/NonCovid) inside splitted folder:
```python
split_to_folders.py
```
To split images into a dictionary:
```python
split_into_dictionary.py
```
To split images into a torch Dataloader:
```python
split_into_dataloader.py
```


