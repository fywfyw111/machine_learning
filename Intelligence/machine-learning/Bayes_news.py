# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 08:56:37 2019

@author: lenovo
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def decision_iris():
    """
    用决策树对鸢尾花分类
    """
    #1)获取数据集
    iris=load_iris()
    #2）划分数据集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)
    #3）决策树预估器
    estimator=DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train,y_train)
    #4)决策评估 
     #方法一：直接对比真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接对比结果：\n",y_predict==y_test)
    #方法二：直接计算准确率
    score=estimator.score(x_test,y_test)
    print("准确率为：\n",score)
    return None
if __name__=="__main__":
    decision_iris()