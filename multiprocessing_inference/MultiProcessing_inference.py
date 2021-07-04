
import numpy as np
import multiprocessing
import time
import ctypes
import os
import keras as k

# os.environ["CUDA_VISIBLE_DEVICES"] = ''

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def createModels(models_list):
    # import keras as k
    for m in models_list:
        model = k.models.Sequential()
        model.add(k.layers.Dense(256, input_shape=(2,)))
        model.add(k.layers.Dense(1, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.save(m)


def get_single_model(model_path):
    # import keras as k
    return k.models.load_model(model_path)


def prediction(model_name):
    # import keras as k
    model = k.models.load_model(model_name)
    ret_val = model.predict(input_c).tolist()[0]
    return ret_val


if __name__ == "__main__":
    # models = ['model1.h5']
    models_list = ['model1.h5', 'model2.h5',
                   'model3.h5', 'model4.h5', 'model5.h5']
    curr_dir = os.listdir(".")
    if models_list[0] not in curr_dir:
        createModels(models_list)
    # Shared array input
    ub = 100
    x_train = np.random.random((5000, 2))
    testShape = x_train[:ub].shape
    input_base = multiprocessing.Array(ctypes.c_double,
                                       int(np.prod(testShape)), lock=False)
    input_c = np.ctypeslib.as_array(input_base)
    input_c = input_c.reshape(testShape)
    input_c[:ub] = x_train[:ub]

    with multiprocessing.Pool() as p:  # Use me for python 3
        # p = multiprocessing.Pool() #Use me for python 2.7
        start_time = time.time()
        res = p.map(prediction, models_list)
        # p.close()  # Use me for python 2.7
        print('Total time taken: {}'.format(time.time() - start_time))
        print(res)

    m = get_single_model(models_list[0])
    start_time = time.time()
    for i in range(len(models_list)):
        m.predict(input_c)
    print('Single Process: {}'.format(time.time() - start_time))
