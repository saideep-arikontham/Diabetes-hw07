import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from readit import read_data


df, y = read_data()

f1 = ['bmi', 's5', 'bp', 's4', 's3', 's6', 's1', 'age', 's2', 'sex']   #r2_score sorted order
f2 = ['bmi', 's5', 'bp', 's3', 'sex', 's1', 's2', 's4', 'age', 's6']  #seq feature selection order
#f3 = ['bmi', 's5', 'bp', 's3', 'sex', 's6', 's1', 's4', 's2', 'age']  #lasso

s1 = []
s2 = []
#s3 = []

for i in range(0,len(f2)-1):
    model1 = LinearRegression()
    model2 = LinearRegression()
    #model3 = LinearRegression()
    X1 = df[f1[:i+1]]
    X2 = df[f2[:i+1]]
    #X3 = df[f3[:i+1]]
    model1.fit(X1,y)
    model2.fit(X2,y)
    #model3.fit(X3,y)

    s1.append(r2_score(y,model1.predict(X1)))
    s2.append(r2_score(y,model2.predict(X2)))
    #s3.append(r2_score(y,model3.predict(X3)))

plt.plot(range(1,10),s1, label = 'sorted r2_scores', marker='o')
plt.legend()

plt.plot(range(1,10),s2, label = 'SequentialFeatureSelector scores', marker='x')
plt.legend()

#plt.plot(range(1,10),s3, label = 'lasso order scores', marker='>')
#plt.legend()

plt.xlabel('n_features')
plt.ylabel('r2_score')

plt.title("Comparing r2_scores of Sorted features and SequentialFeatureSelector features", fontsize = 10)
plt.savefig('figs/difference.png')
plt.show()
