import pandas as pd


df=pd.read_csv('survey.csv')

df=df[(df['Age'] > 18) & (df['Age'] <80)]
df['Gender'] = df['Gender'].str.lower().str.strip()
map = {'m':'Male','male':'Male','maile':'Male','something kinda male?':'Male','cis male':'Other','mal':'Male','male (cis)':'Male','queer/she/they':'Female','make':'Male','guy (-ish) ^_^':'Other',   
'man':'Male','male leaning androgynous':'Other','malr':'Male','cis man':'Other','mail':'Male','nah':'Other','all':'Unknown','fluid':'Other','p':'Other','neuter':'Other','a little about you':'Other'}
df['Gender']= df['Gender'].map(map)
df['Gender']=df['Gender'].fillna('Female')
df['state']=df['state'].fillna('Unknown')
df['self_employed']=df['self_employed'].fillna('Unknown')
df['work_interfere']=df['work_interfere'].fillna('Unknown')

map_employees={'6-25': 1,'More than 1000' :5 ,'26-100' :2,'100-500' :3,'1-5':0, '500-1000':4}  
df['no_employees']=df['no_employees'].map(map_employees)

df.to_csv('cleaned_survey.csv', index=False)
