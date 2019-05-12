import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report, recall_score,precision_score,precision_recall_curve,average_precision_score
warnings.filterwarnings('ignore')
path = os.getcwd()
print(path)
path = path.strip('src')
os.chdir(path)



def missing_values(dataframe):
    missing = pd.DataFrame({'total_missing': dataframe.isnull().sum(),
                            'percent_missing': (dataframe.isnull().sum() / dataframe.shape[0]) * 100})
    return missing


def splitnumcat(dataframe):
    numAttr = dataframe.select_dtypes(include=['float64', 'int64']).columns
    catAttr = dataframe.select_dtypes(exclude=['float64', 'int64']).columns
    return numAttr, catAttr


def unique_values(dataframe):
    percentage_unique = pd.DataFrame((dataframe.nunique() / dataframe.shape[0]) * 100)
    return percentage_unique


def correlation_df(dataframe):
    corr = dataframe.corr(method='pearson')
    corr_uns = corr.unstack()
    corr_uns_sort = pd.DataFrame(corr_uns.sort_values(ascending=False, kind="quicksort")).reset_index().rename(
        {'level_0': 'col_1', 'level_1': 'col_2', 0: 'corr'}, axis='columns')
    corr_uns_sort = corr_uns_sort[corr_uns_sort['col_1'] != corr_uns_sort['col_2']]
    corr_uns_sort = corr_uns_sort.dropna()
    corr_uns_sort = corr_uns_sort.sort_values(by=['corr'], ascending=False)
    return corr_uns_sort

def plot_precision_recall_vs_threshold(y_test,y_test_pred_prob,path):
    y_test_pred_prob_1 = y_test_pred_prob[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_pred_prob_1,pos_label='yes')
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.savefig(path+'/visualization/precision-recall-threshold.png')


def printTree(clf):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph.write_png('tree.png')


class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns  # list of column to encode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''

        output = X.copy()

        if self.columns is not None:
            for col in self.columns:
                print(col)
                output[col] = LabelEncoder().fit_transform(output[col])
           
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


print("ran all functions")

