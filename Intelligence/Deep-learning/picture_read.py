# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:44:31 2019
狗图片读取案例
@author: lenovo
"""
import tensorflow as tf
import os
def picture_read(file_list):
    #1)构造文件名队列
    file_queue=tf.train.string_input_producer(file_list)
    #2)读取与解码
    #读取阶段
    reader=tf.WholeFileReader()
    #key是文件名，value是一张图片的原始编码形式
    key,value=reader.read(file_queue)
    print("key:\n",key)
    print("value:\n",value)
    #解码阶段
    image=tf.image.decode_jpeg(value)
    print("image:\n",image)
    #图像的形状类型修改
    image_resize=tf.image.resize_images(image,[200,200])
    #静态形状修改
    image_resize.set_shape(shape=[200,200,3])
    #3)批处理队列
    image_batch=tf.train.batch([image_resize],batch_size=10,num_threads=1,capacity=10)
    print("image_batch:\n",image_batch)
    #开启会话
    with tf.Session() as sess:
        #开启线程
        #线程协调员
        coord=tf.train.Coordinator()
        thread=tf.train.start_queue_runners(sess=sess,coord=coord)
        new_key,new_value,new_image,new_batch=sess.run([key,value,image,image_resize])
        #print("new_key:\n",new_key)
        #print("new_value:\n",new_value)
        print("new_image:\n",new_image)
        print("new_batch:\n",new_batch)
        #回收线程
        coord.request_stop()
        coord.join(thread)
    return None
if __name__=="__main__":
    #构造路径加文件名的列表
    filename=os.listdir("C:/Users/lenovo/Desktop/Dog")
    #拼接路径加文件名
    file_list=[os.path.join("C:/Users/lenovo/Desktop/Dog/",file )for file in filename]
    picture_read(file_list)