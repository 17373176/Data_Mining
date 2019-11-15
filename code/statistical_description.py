# ****数据预处理代码-第二步：抽样描述统计**** #

import numpy as NP
import pandas as PD
from PIL import Image
import matplotlib.pyplot as PLT
from collections import Counter
import seaborn as sns

# 正则化


'''from sklearn.preprocessing import Normalizer

z = pd.DataFrame({"a": [2., 1., 6.], "b": [3., 0, 2.]})
Normalizer.fit_transform(z)
a = Normalizer()
a.fit(z)
a.transform(z)'''

# *一、数据描述统计* #

# 1.制表法

# 2.绘图法 Matplotlib

'''准备数据'''
x = ['A', 'B', 'C', 'D', 'E']
y = [1, 2, 3, 4, 5]
'''构图'''
'''PLT.bar(x, y)  # 条形图
PLT.savefig('./pictureBar.png')
PLT.show()

nums = [31, 66, 10, 9]
labels = ['high', 'big', 'middle', 'small']
PLT.pie(x=nums, labels=labels)  # 饼图
PLT.savefig('./picturePie.png')  # 存储图片
PLT.show()  # 展示绘图
'''
'''
img = Image.open('./img.png')
print(img.size)
'''

# 3.数值法:平均值、中位数、离散程度：方差、极差、Z分数（观测值-均值）/标准差、正态分布曲线，可利用describe()返回大量指标

'''创建Dataframe数据库'''
'''df = PD.DataFrame(NP.arange(12, 32).reshape((5, 4)), index=["a", "b", "c", "d", "e"], columns=["WW", "XX", "YY", "ZZ"])
也可以这样构造 df = PD.Series([1, 2])'''
df = PD.Series([1, 1, 1, 1, 1])

# print(df.var())  # 方差
# print(df)  # 打印数据表
# print(df.mean())  # 每一列的平均值
# print(df['YY'].mean())  # 指定列的平均值
# print(df['YY'])  # Series类型，输出指定列
# print(df.max())  # 每一列的最大值 Series类型。  min()最小值
# argmax()某一列最大值的位置  argmin()最小值的位置
# print(df['YY'].argmax())  # e
# print(df['YY'].median())  # 中位数
# print(df['YY'].std())  # 标准差

