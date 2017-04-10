import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, confusion_matrix


result_log = open('results/log.md', 'w')


df = pd.read_csv('camel-1.6.csv',
                 usecols=[3, 4, 5, 6, 7, 8, 23],
                 dtype={'wmc': np.float32, 'dit': np.float32,
                        'noc': np.float32, 'cbo': np.float32,
                        'rfc': np.float32, 'lcom': np.float32},
                 converters={23: lambda x: 1 if x > '0' else 0})

X = df.loc[:, ('wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom')].values
y = df['bug'].values.reshape((-1, 1))

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

kf = KFold(n_splits=10, shuffle=True, random_state=33)

n_fold = 1

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    result_log.write("# FOLD %02d\n" % n_fold)

    model = Sequential()
    model.add(Dense(50, input_dim=6, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(70, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(90, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, epochs=1000, batch_size=10, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = model.predict_classes(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.savefig("results/roc/fold_%02d" % n_fold)


    result_log.write("##### Accuracy : %f\n" % (accuracy_score(y_test, y_pred_classes)))
    result_log.write("##### Precision : %f\n" % (precision_score(y_test, y_pred_classes)))
    result_log.write("##### Recall : %f\n" % (recall_score(y_test, y_pred_classes)))

    result_log.write("### Confusion Matrix\n")
    cm = confusion_matrix(y_test, y_pred_classes, labels=[0, 1])

    result_log.write("|       |    0    |    1    |\n")
    result_log.write("|-------|---------|---------|\n")
    result_log.write("|   0   |{:^9}|{:^9}|\n".format(cm[0, 0], cm[0, 1]))
    result_log.write("|   1   |{:^9}|{:^9}|\n".format(cm[1, 0], cm[1, 1]))

    result_log.write("### ROC Curve\n")
    result_log.write("![](results/roc/fold_%02d.png)\n" % n_fold)

    result_log.write("##### AUC : %f\n" % roc_auc)


    n_fold += 1
    result_log.write("\n---\n\n")

result_log.close()
