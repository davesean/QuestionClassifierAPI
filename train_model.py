import pandas
from model import QuestionClassifier
from sklearn.model_selection import train_test_split
import numpy as np

df = pandas.read_csv('aufgabe_mle.csv')
data = df.as_matrix(columns=df.columns[1:])

X_train_big, X_test = train_test_split(data, test_size=0.1, random_state=42)
X_train, X_val = train_test_split(X_train_big, test_size=0.05, random_state=42)

features = X_train[:, :2]
labels = X_train[:, 2]

model = QuestionClassifier(num_features=features.shape[-1], num_nodes=8)
model.fit(features, labels, 2000, 1000, (X_val[:, :2], X_val[:, 2]))
loss = model.predict_eval(X_test[:, :2], X_test[:, 2])
print("Test loss: %f" % loss)
pred, prob = model.predict(X_test[:, :2])
print(np.sum((X_test[:, 2] + pred[:, 0]) == 0), np.sum(X_test[:, 2] == 0))
