import sys
import os
sys.path.append('../')
from prediction.predict import get_prediction
from training.train import get_classifier, train_model
from data_IO.data_reader import get_data
from statistic.calculation import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCH = 100
DATA_PATH = '/home/liuyajun/concat_net/data/yolo_motion/'
for k in range(0, 5):
    MODEL_PATH = '/home/liuyajun/concat_net/model/yolo_motion/k' + str(k) + '/'
    TRAIN_LABEL_PATH = '/home/liuyajun/concat_net/data/kfold_label/train_k' + str(k) + '_label.csv'
    TEST_LABEL_PATH = '/home/liuyajun/concat_net/data/kfold_label/test_k' + str(k) + '_label.csv'

    clf = get_classifier(MODEL_PATH, 'alex_net')
    if not os.path.exists(MODEL_PATH):
        print('Training k%i ....' % k)
        train_data, train_label, _ = get_data(TRAIN_LABEL_PATH, DATA_PATH)
        for i in range(EPOCH):
            train_model(clf, {'X1': train_data}, train_label)

    test_data, test_label, _ = get_data(TEST_LABEL_PATH, DATA_PATH)
    pred = get_prediction(clf, {'X1': test_data}, test_label)
    print('Result of k%i:' % k)
    confusion_matrix(pred, test_label, show_mat=True)
    del clf