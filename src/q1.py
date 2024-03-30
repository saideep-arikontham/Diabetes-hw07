import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from readit import read_data, get_r2_scores

df, y = read_data()

feature, score = get_r2_scores(df, y, df.columns)
scores_df = pd.DataFrame(data = {'feature':feature, 'r2_score':score})
final = scores_df.sort_values(by = 'r2_score', ascending = False)
sns.barplot(data = final, x = 'feature', y = 'r2_score')
plt.title('Sorted r2 scores for Features in Diabetes data')
plt.savefig('figs/sorted_feature_scores.png')
plt.show()
