'''
===================================================
Keras prediction with MultiThread
===================================================
EXPLANATION:
	Keras predict方法是線程安全的
    官方文件並沒有寫到，但是這裡有人提問，本script就是這裡的code
    https://stackoverflow.com/questions/46725323/keras-tensorflow-exception-while-predicting-from-multiple-threads
    強化學習中有一個叫做A3C的演算法，也需要用到MultiThread，這裡有解釋
    https://stackoverflow.com/questions/40850089/is-keras-thread-safe
    拿到MultiThread的return
    https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python

INPUTS:
OPTIONAL INPUT:
OUTPUT:
EXAMPLE:
	python MultiThread_inference.py
REVISION HISTORY:
	functionality1  date1 author1
	functionality2  date2 author2
'''
print(__doc__)
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import threading as t
import numpy as np
import time
import os

from tensorflow.python.ops.gen_math_ops import mul
K.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # if you want to run usingh CPU


class CNN:
    def __init__(self, model_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            self.cnn_model = load_model(model_path)
            for _ in range(5):
                self.cnn_model.predict(np.array([[0, 0]]))  # warmup
            init = tf.global_variables_initializer()
            self.session.run(init)
        self.cnn_model._make_predict_function()
        print(
            f'session id : {id(self.session)}, graph id : {id(self.graph)}')
        # self.cnn_model._make_predict_function()  # make it thread safe
        # self.graph.finalize()  # finalize

    def preproccesing(self, data):
        # dummy
        return data

    def query_cnn(self, data):
        X = self.preproccesing(data)
        with self.graph.as_default():
            with tf.Session(graph=self.graph):
                prediction = self.cnn_model.predict(X)
        # print(prediction)
        return prediction


cnn = CNN("keras_note/keras_multi/dummymodel")
cnn2 = CNN("keras_note/keras_multi/dummymodel")
cnn3 = CNN("keras_note/keras_multi/dummymodel")
cnn4 = CNN("keras_note/keras_multi/dummymodel")
cnn5 = CNN("keras_note/keras_multi/dummymodel")
t_base = time.time()
cnn.query_cnn(np.random.random((5000, 2)))
print('Baseline 1 thread 1 time: ', time.time() - t_base)

mul_t = time.time()
# 可以寫一個 for loop，cycle，或是fit(X_)
th = t.Thread(target=cnn.query_cnn, kwargs={
              "data": np.random.random((5000, 2))})
th2 = t.Thread(target=cnn2.query_cnn, kwargs={
               "data": np.random.random((5000, 2))})
th3 = t.Thread(target=cnn3.query_cnn, kwargs={
               "data": np.random.random((5000, 2))})
th4 = t.Thread(target=cnn4.query_cnn, kwargs={
               "data": np.random.random((5000, 2))})
th5 = t.Thread(target=cnn5.query_cnn, kwargs={
               "data": np.random.random((5000, 2))})
th.start()
th2.start()
th3.start()
th4.start()
th5.start()

th2.join()
th.join()
th3.join()
th5.join()
th4.join()
print('5 Thread 5 models, 1 time : ', time.time() - mul_t)

sin_t = time.time()
cnn.query_cnn(np.random.random((25000, 2)))
print('1 Thread, concat each X to 5 times large : ', time.time() - sin_t)
