import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from readit import read_data, get_r2_scores


X, y = read_data()

# Create a LinearRegression model (you can use any other model)
clf = LinearRegression()

# Create a SequentialFeatureSelector

final_order = []
for i in range(1,len(X.columns)):
    sfs = SequentialFeatureSelector(clf, n_features_to_select=i, direction='forward')
    sfs.fit(X, y)
    selected_features = set(X.columns[sfs.get_support()])
    final_order.append(list(set(final_order) ^ selected_features)[0])
    prev = selected_features
    
final_order.append(list(set(final_order) ^ set(X.columns))[0])
print(final_order)


feature, score = get_r2_scores(X, y, final_order)
scores_df = pd.DataFrame(data = {'feature':feature, 'r2_score':score})
sns.barplot(data = scores_df, x = 'feature', y = 'r2_score')
plt.title("SequentialFeatureSelector feature order")
plt.savefig('figs/seq_feature_selector.png')
plt.show()
