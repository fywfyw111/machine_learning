# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:40:49 2019

@author: lenovo
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def Liner_regression():
    """"
    自实现一个线性回归
    """
    #增加命名空间with tf.variable_scope("prepare_data):
    #1)准备数据
    X=tf.random_normal(shape=[100,1])
    y_true=tf.matmul(X,[[0.8]])+0.7
    #2)构造模型
    #定义模型参数用变量
    weights=tf.Variable(initial_value=tf.random_normal(shape=[1,1]))
    bias=tf.Variable(initial_value=tf.random_normal(shape=[1,1]))
    y_predict=tf.matmul(X,weights)+bias
    #3)损失函数
    #均方误差
    error=tf.reduce_mean(tf.square(y_true-y_predict))
    #4)优化损失
    optimazor=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    #2-收集变量
    tf.summary.scalar("error",error)#收集标量
    tf.summary.histogram("weights",weights)
    tf.summary.histogram("bias",bias)
    #3-合并变量
    merged=tf.summary.merge_all()
    #创建server对象
    saver=tf.train.Saver()
    #显示的初始化变量
    init=tf.global_variables_initializer()
    #开启会话
    with tf.Session() as sess:
        sess.run(init)
        #1-创建事件文件
        file_writer=tf.summary.FileWriter("c:/Users/lenovo/Desktop/test",graph=sess.graph)
        
        #查看训练前模型参数
        print("训练前模型参数为：权重%f,偏置%f,损失%f" % (weights.eval(),bias.eval(),error.eval()))
        #开始训练
        """
        for i in range(100):
            sess.run(optimazor)
            #4-每次迭代都运行合并
            summary=sess.run(merged)
            #5-将每次迭代后的数据写入事件文件
            file_writer.add_summary(summary,i)
            #保存模型
            if i%10==0:
               saver.save(sess,"c:/Users/lenovo/Desktop/model/my_liner.ckpt")
        print("训练后模型参数为：权重%f,偏置%f,损失%f" % (weights.eval(),bias.eval(),error.eval()))
        """
        #加载模型
        if os.path.exists("c:/Users/lenovo/Desktop/model/checkpoint"):
            saver.restore(sess,"c:/Users/lenovo/Desktop/model/my_liner.ckpt")
        print("训练后模型参数为：权重%f,偏置%f,损失%f" % (weights.eval(),bias.eval(),error.eval()))
    return None
if __name__=="__main__":
    #代码1 TensorFlow的基本结构
    Liner_regression()
