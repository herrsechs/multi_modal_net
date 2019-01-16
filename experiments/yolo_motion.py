import sys
import os
sys.path.append('../')
from prediction.predict import get_prediction
from training.train import get_classifier
from data_IO.data_reader import get_data
from statistic.calculation import confusion_matrix

LABEL_PATH = '/home/liuyajun/concat_net/data/kfold_label/test_k0_label.csv'
DATA_PATH = '/home/liuyajun/concat_net/data/yolo_motion/'
MODEL_PATH = '/home/liuyajun/concat_net/model/yolo_motion/k0/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

labels, imgs, _ = get_data(DATA_PATH, LABEL_PATH)
clf = get_classifier(MODEL_PATH, 'alex_net')
pred = get_prediction(clf, {'X1': imgs}, labels)
confusion_matrix(pred, labels, show_mat=True)
