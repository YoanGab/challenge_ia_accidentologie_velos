from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss

import matplotlib.pyplot as plt
import seaborn as sns

def eval_model(y_test, y_pred, name):
    print("="*30)
    print(name)
    print('****Results****')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()