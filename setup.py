import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="patient_aware_splitter", # Replace with your own username
    version="0.0.1",
    author="Maede Maftouni",
    author_email="maftouni@vt.edu",
    description="This package splits a medical imaging dataset into test and train sets in a patient aware and stratified manner.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maftouni/Patient_Aware_Splitter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)