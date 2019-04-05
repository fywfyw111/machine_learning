# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:03:27 2019

@author: lenovo
"""
"""
使用正规方程和梯度下降来对波士顿房价进行预测
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def Ridege_boston():
    """
    岭回归对波士顿房价进行预测
    """
    #1)获取数据
    boston=load_boston()
    #2)划分数据集
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)
    #3)特征工程：标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    #4)预估器流程
    estimator=Ridge(max_iter=10000,alpha=0.5)
    estimator.fit(x_train,y_train)
    #5)得到模型
    print("岭回归权重系数：\n",estimator.coef_)
    print("岭回归偏置为:\n",estimator.intercept_)
    #6)模型评估
    y_predict=estimator.predict(x_test)
    error=mean_squared_error(y_test,y_predict)
    print("岭回归-均方误差为：\n",error)
    return None    
if __name__=="__main__":
   Ridege_boston()
    