# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:04:08 2019

@author: lenovo
"""
"""
使用逻辑回归对数据进行初始化
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
def logic_cancer():
    #1.读取数据
    #path="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin//breast-cancer-wisconsin.data"
    column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin//breast-cancer-wisconsin.data',names=column_names)
    #2.缺失值处理
    #替换成np.nan
    data.replace(to_replace="?",value=np.nan)
    #删除缺失样本
    data.dropna(inplace=True)
    #3.划分数据集
    #筛选特征值和目标值
    x=data.iloc[:,1:-1]#行全要。列从1到倒数第二行
    y=data["Class"]
    x_train,x_test,y_train,y_test=train_test_split(x,y)
    #4.标准化
    transfer=StandardScaler()
    transfer.fit_transform(x_train)
    transfer.transform(x_test)
    #5.逻辑回归预估器流程
    estimator=LogisticRegression()
    estimator.fit(x_train,y_train)
    #回归系数和偏置
    coef=estimator.coef_
    print("回归系数：\n",coef)
    print("偏置：\n",estimator.intercept_)
    #6模型评估
     #方法一：直接对比真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接对比结果：\n",y_predict==y_test)
    #方法二：直接计算准确率
    score=estimator.score(x_test,y_test)
    print("准确率为：\n",score)
if __name__=="__main__":
    logic_cancer()
