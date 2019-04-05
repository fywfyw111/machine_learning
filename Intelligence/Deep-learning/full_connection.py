# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:49:51 2019
手写数字识别——全连接
@author: lenovo
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def full_connection():
    #1准备数据 用占位符占位
    mnist=input_data.read_data_sets("./mnist_data",one_hot=True)#存储数据，目标值用one-hot编码
    x=tf.placeholder(dtype=tf.float32,shape=(None,784))#占位指定类型和形状
    y_true=tf.placeholder(dtype=tf.float32,shape=(None,10))
    
    #2构造模型 模型参数用变量存储 要指定
    weights=tf.Variable(initial_value=tf.random.normal(shape=[784,10]))
    bias=tf.Variable(initial_value=tf.random.normal(shape=[10]))
    y_predict=tf.matmul(x,weights)+bias
    #3构造损失函数
    error=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
    #4优化损失
    optimazer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    #5准确率计算
    equal_list=tf.equal(tf.argmax(y_true,1),tf.arg_max(y_predict,1))
    accuracy=tf.reduce_mean(tf.cast(equal_list,tf.float32))
    #初始化变量
    init=tf.global_variables_initializer()
    #开启会话
    with tf.Session() as sess:
        sess.run(init)
        image,label=mnist.train.next_batch(100)#获取一个批次的样本大小为100
        print("训练前损失为%f" % sess.run(error,feed_dict={x:image,y_true:label}))
        #开始训练
        for i in range(1000):
            _,loss,accuracy_value=sess.run([optimazer,error,accuracy],feed_dict={x:image,y_true:label})
            print("第%d次训练，损失为%f,准确率为%f"%(i+1,loss,accuracy_value))
    return None
    
    
if __name__=="__main__":
    full_connection()