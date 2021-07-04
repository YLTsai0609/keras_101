import tensorflow as tf

graph = tf.get_default_graph()
g1 = tf.Graph()  # 加载到Session 1的graph
g2 = tf.Graph()  # 加载到Session 2的graph

sess1 = tf.Session(graph=g1)  # Session1
sess2 = tf.Session(graph=g2)  # Session2

# 加载第一个模型
with sess1.as_default():
    assert tf.get_default_graph() is graph
    with g1.as_default():
        assert tf.get_default_graph() is g1
# 加载第二个模型
with sess2.as_default():  # 1
    with g2.as_default():
        assert tf.get_default_graph() is g2

with sess1.as_default():
    with sess1.graph.as_default():  # 2
        assert tf.get_default_graph() is g1

with sess2.as_default():
    assert tf.get_default_graph() is graph
    with sess2.graph.as_default():
        assert tf.get_default_graph() is g2
# 关闭sess
sess1.close()
sess2.close()

print('done')
