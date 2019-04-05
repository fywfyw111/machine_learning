# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
def knn_iris_cv():
    """
    用KNN算法对鸢尾花进行分类,使用交叉验证
    """
    #1)获取数据
    iris=load_iris()
    #2)划分数据集，不要把全部数据集都拿来训练，一般测试集占20%-30%，默认25%
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)
    #3)特征工程的数据预处理，进行标准化
    transfer=StandardScaler()
    #这里训练集正常标准化，到那时测试集要注意不能按照自己的平均值标准化，要用训练集的平均值，保证测试的准确性
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    #4)KNN算法预估器
    estimator=KNeighborsClassifier()
    #加入网格搜索与交叉验证
    #参数准备
    para_id={"n_neighbors":[1,3,5,7,9]}
    estimator= GridSearchCV(estimator,param_grid=para_id,cv=10)
    estimator.fit(x_train,y_train)
    #5)模型评估，两种方法
    #方法一：直接对比真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接对比结果：\n",y_predict==y_test)
    #方法二：直接计算准确率
    score=estimator.score(x_test,y_test)
    print("准确率为：\n",score)
    #最佳参数
    print("最佳参数：\n",estimator.best_params_)
    #最佳结果
    print("最佳结果：\n",estimator.best_score_)
    return None

if __name__=="__main__":
    knn_iris_cv()
