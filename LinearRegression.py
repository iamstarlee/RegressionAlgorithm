import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import metrics

sns.set_style('white',{'font.sans-serif':['simhei','Arial']})  #解决中文不能显示问题

iris=datasets.load_iris()
iris_data= pd.DataFrame(iris.data,columns=iris.feature_names)
iris_data['species']=iris.target_names[iris.target]
iris_data.head(3).append(iris_data.tail(3))   #前面三条+后面三条
iris_data.rename(columns={"sepal length (cm)":"萼片长",
                     "sepal width (cm)":"萼片宽",
                     "petal length (cm)":"花瓣长",
                     "petal width (cm)":"花瓣宽",
                     "species":"种类"},inplace=True)
kind_dict = {
    "setosa":"山鸢尾",
    "versicolor":"杂色鸢尾",
    "virginica":"维吉尼亚鸢尾"
}
iris_data["种类"] = iris_data["种类"].map(kind_dict)

#g = sns.pairplot(iris_data)

#hue：针对某一字段进行分类
#g = sns.pairplot(iris_data,hue='种类')

#kind:用于控制非对角线上图的类型，可选'scatter'与'reg'
#diag_kind:用于控制对角线上的图分类型，可选'hist'与'kde'

#g = sns.pairplot(iris_data,kind='reg',diag_kind='ked')
g = sns.pairplot(iris_data,kind='reg',diag_kind='hist')

#vars：研究某2个或者多个变量之间的关系vars,
#x_vars,y_vars：选择数据中的特定字段，以list形式传入需要注意的是，x_vars和y_vars要同时指定

#g = sns.pairplot(iris_data,vars=["萼片长","花瓣长"])
g = sns.pairplot(iris_data,x_vars=["萼片长","花瓣宽"],y_vars=["萼片宽","花瓣长"])
plt.show()