import pandas as pd
df1 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/buisness.csv')
df2 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/coronavirus.csv')
df3 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/dispora.csv')
df4 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/entertainment.csv')
df5 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/features.csv')
df6 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/ipl2020.csv')
df7 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/nation.csv')
df8 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/news.csv')
df9 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/news1.csv')
df10 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Desktop/data science/news_recommender_system/quotetutorials/sports.csv')
df11 = pd.read_csv('C:/Users/ARSHDEEP SINGH/Downloads/updated_Final_Report.csv')
df11 = df11.rename(columns = {'date': 'time'})

frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]
result = pd.concat(frames)
result.shape
result
result.to_csv('total.csv',index = False)
