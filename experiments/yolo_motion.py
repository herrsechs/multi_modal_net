from prediction.predict import get_prediction
from training.train import get_classifier
from data_IO.data_reader import get_data
from statistic.calculation import confusion_matrix

LABEL_PATH = './label.txt'
DATA_PATH = './data'
MODEL_PATH = './model'

labels, imgs, _ = get_data(DATA_PATH, LABEL_PATH)
clf = get_classifier(MODEL_PATH, 'alex_net')
pred = get_prediction(clf, {'X': imgs}, labels)
confusion_matrix(pred, labels, show_mat=True)
