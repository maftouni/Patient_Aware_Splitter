df = pd.read_csv('sample_meta_data.csv')
df['Patient ID'] = df['Patient ID'].astype(str)


#folders
splitter(df,'Patient ID','COVID-19 Infection')

# dictionary
splitted_dictionary = splitter(df,'Patient ID','COVID-19 Infection')
