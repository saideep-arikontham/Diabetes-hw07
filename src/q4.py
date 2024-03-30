import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from readit import read_data

X, y = read_data()

pcr = make_pipeline(StandardScaler(), PCA(), LinearRegression())
n_components = np.arange(1, X.shape[1]+1)
n_folds=5

clf = GridSearchCV(pcr, {'pca__n_components': n_components}, cv=n_folds,
                   refit=True, return_train_score=True)
clf.fit(X, y)

scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]
std_error = scores_std / np.sqrt(n_folds)

train_scores = clf.cv_results_["mean_train_score"]
train_scores_std = clf.cv_results_["std_train_score"]
train_std_error = train_scores_std / np.sqrt(n_folds)

plt.figure().set_size_inches(8, 6)

plt.plot(n_components, train_scores, color="tab:blue", label=f'training score (mean = {np.mean(train_scores):.3f} +/- {np.mean(train_std_error):.3f})')
plt.plot(n_components, train_scores + train_std_error, linestyle="--", color="tab:blue")
plt.plot(n_components, train_scores - train_std_error, linestyle="--", color="tab:blue")
plt.fill_between(n_components, train_scores + train_std_error, train_scores - train_std_error, alpha=0.2)


plt.plot(n_components, scores, color="tab:orange", label=f"test score (mean = {np.mean(scores):.3f} +/- {np.mean(std_error):.3f})")
plt.plot(n_components, scores + std_error, linestyle="--", color="tab:orange")
plt.plot(n_components, scores - std_error, linestyle="--", color="tab:orange")
plt.fill_between(n_components, scores + std_error, scores - std_error, color="tab:orange", alpha=0.2)

plt.ylabel("CV score +/- std error")
plt.xlabel("n_components")
plt.title("PCR with cross-validation")
plt.legend()
plt.axhline(np.max(scores), linestyle=":", color=".5")

plt.savefig('figs/pcr.png')
plt.show()
