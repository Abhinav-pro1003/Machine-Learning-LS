import pandas as pd
import numpy as np

data = {
    'Name' : ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Raj', 'Kamal', 'Suriya', 'Prakash', 'Deepak'],
    'Subject' : ['Math', 'Science', 'English', 'Math', 'Science', 'English', 'Math', 'Science', 'English', 'Math'],
    'Score' : np.random.randint(50, 100, size=10).tolist(),
    'Grade' : ['None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None']
}

df=pd.DataFrame(data)
df.loc[df['Score'] < 60, 'Grade'] = 'F'
df.loc[df['Score'] >= 60, 'Grade'] = 'D'
df.loc[df['Score'] >= 70, 'Grade'] = 'C'
df.loc[df['Score'] >= 80, 'Grade'] = 'B'
df.loc[df['Score'] >= 90, 'Grade'] = 'A'

df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

print(df_sorted)

print()

avg_by_subject = df.groupby('Subject')['Score'].mean()
print("Average score by subject:")
print(avg_by_subject)

print()

def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A', 'B'])].reset_index(drop=True)

print("Students with grade A or B:\n", pandas_filter_pass(df_sorted))