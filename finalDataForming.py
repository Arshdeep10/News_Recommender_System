import pandas as pd

article_data = pd.read_csv('cluter_report.csv')
user_data = pd.read_csv('Generated_Click_Stream_Data.csv')

article_id = user_data['ArticleId_served'].unique()
print(len(article_id))

article_data['ArticleId_served'] = article_id
article_data.head()

final_data = pd.merge(article_data, user_data, on='ArticleId_served')
lenght_of_content = []
for i in final_data['content']:
    i = i.split()
    lenght_of_content.append(len(i))

final_data['lenght_content'] = lenght_of_content

rating = ((final_data['Percentage Time Spent'] * 3)/final_data['lenght_content'])*10
final_data['Rating'] = rating
final_data.columns

for index in range(len(final_data)):
    if final_data['Rating'][index] > 10.0:
        final_data['Rating'][index] = 10
        

final_data.to_csv('finaldata.csv')

