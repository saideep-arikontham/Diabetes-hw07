from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def read_data():
    return load_diabetes(return_X_y = True, as_frame = True)


def get_r2_scores(df, y, x_cols):
    score = []
    feature = []
    for col in x_cols:
        X = df[[col]]

        model = LinearRegression()
        model.fit(X,y)
        feature.append(col)
        r2 = r2_score(y,model.predict(X))
        score.append(r2)
        #print(col,r2)
    return feature, score