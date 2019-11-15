# ****数据预处理代码-第一步：数据清洗**** #

import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
import missingno
from scipy import stats
import math

# 绘图预处理
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 预备数据
list_prov_code = {  # <province, adcode>
    '河南': '410000', '广东': '440000', '山东': '370000', '河北': '130000', '江苏': '320000', '四川': '510000',
    '上海': '310000', '云南': '530000', '陕西': '610000', '山西': '140000', '广西': '450000', '重庆': '150000',
    '内蒙古': '320000', '湖南': '430000', '北京': '110000', '安徽': '340000','辽宁': '210000',
    '黑龙江': '230000', '江西': '360000', '福建': '350000', '浙江': '330000', '湖北': '420000'
}

list_season = {'season1-2016': 0, 'season2-2016': 0, 'season3-2016': 0, 'season4-2016': 0,
               'season1-2017': 0, 'season2-2017': 0, 'season3-2017': 0, 'season4-2017': 0}

list_province = list_prov_code.keys()  # [province]

list_adcode = list_prov_code.values()  # [adcode]

list_model = []  # [model] 通过遍历数据添加

dict_model_int = dict()  # <model, int> 车型的数据映射

list_bodyType = ['SUV', 'Sedan', 'MPV', 'Hatchback']  # [bodyType]

sum_sales = 0  # 销量总和

# 自定义函数


def is_digit(num):  # 判断是否为数值
    if isinstance(num, np.int64) or isinstance(num, float) or isinstance(num, int) and not math.isnan(num):
        return True
    return False


def is_str(string, this_list):  # 判断是否为合理字符串
    if isinstance(string, str):
        if str in this_list:
            return True
    return False


def add_model(df):  # 获取model数据并映射到dict_model_int
    i = 1
    for model in df['model']:
        if model not in list_model:
            list_model.append(model)
            dict_model_int.update({model: i})
            i += 1


def sales_sum_prov(df):  # 分省份对销量求和，返回<省,销量>
    prov_data = dict()
    list_str = list_province
    for str in list_str:
        for i in range(1, df.shape[0]):
            row_date = df.iloc[i]
            if row_date[0] == str and is_digit(row_date[6]):  # 把异常省份名称和异常销量值、空数据排除
                if str in prov_data.keys():
                    prov_data[str] += row_date[6]
                else:
                    prov_data.update({str: row_date[6]})
    print(prov_data)
    return prov_data


def sales_sum_season(df, str):  # 对某一省份按季度销量求和，返回<季度,销量>
    season_data = dict(list_season)
    for i in range(1, df.shape[0]):
        row_date = df.iloc[i]
        if row_date[0] == str and is_digit(row_date[6]):  # 把异常省份名称和异常销量值排除
            if not (math.isnan(row_date[5]) and math.isnan(row_date[4])):  # 排除年月为空
                if row_date[5] in [1, 2, 3]:  # 找到对应月份的季度
                    if row_date[4] == 2016:
                        season_data['season1-2016'] += row_date[6]
                    else:
                        season_data['season1-2017'] += row_date[6]
                elif row_date[5] in [4, 5, 6]:
                    if row_date[4] == 2016:
                        season_data['season2-2016'] += row_date[6]
                    else:
                        season_data['season2-2017'] += row_date[6]
                elif row_date[5] in [7, 8, 9]:
                    if row_date[4] == 2016:
                        season_data['season3-2016'] += row_date[6]
                    else:
                        season_data['season3-2017'] += row_date[6]
                elif row_date[5] in [10, 11, 12]:
                    if row_date[4] == 2016:
                        season_data['season4-2016'] += row_date[6]
                    else:
                        season_data['season4-2017'] += row_date[6]
    return season_data


def body_type_sales_sum(df):  # 对车身类型销量求和，返回<车身类型，销量>
    global sum_sales  # 声明全局变量
    dict_bodyType_sales = dict({'SUV': 0, 'Sedan': 0, 'MPV': 0, 'Hatchback': 0})
    for str in list_bodyType:
        for i in range(1, df.shape[0]):
            row_date = df.iloc[i]
            if row_date[3] == str and is_digit(row_date[6]):
                dict_bodyType_sales[str] += row_date[6]
                sum_sales += row_date[6]
    return dict_bodyType_sales.values()


def model_sales_sum(df):  # 销售量求和，返回字典<车型，销售量>
    dict_sales_model = dict()
    for str in list_model:
        dict_sales_model.update({dict_model_int[str]: 0})
        for i in range(1, df.shape[0]):
            row_date = df.iloc[i]
            if row_date[2] == str and is_digit(row_date[6]):
                dict_sales_model[dict_model_int[str]] += row_date[6]
    return dict_sales_model


def search_sum_model(df):  # 搜索量求和，返回字典<车型，搜索量>
    dict_search_model = dict()
    for str in list_model:
        dict_search_model.update({dict_model_int[str]: 0})
        for i in range(1, df.shape[0]):
            row_date = df.iloc[i]
            if row_date[2] == str and is_digit(row_date[5]):
                dict_search_model[dict_model_int[str]] += row_date[5]
    return dict_search_model


def comment_sum_model(df):  # 评论量求和，返回字典<车型，评论量>
    dict_comment_model = dict()
    for str in list_model:
        dict_comment_model.update({dict_model_int[str]: 0})
        for i in range(1, df.shape[0]):
            row_date = df.iloc[i]
            if row_date[0] == str and is_digit(row_date[3]):
                dict_comment_model[dict_model_int[str]] += row_date[3]
    return dict_comment_model


def com_rep_sum_season(df, index):  # 对评论量和回复量求和，并绘制对比折线图
    season_data_com = dict(list_season)
    season_data_rep = dict(list_season)
    str = list(dict_model_int.keys())[list(dict_model_int.values()).index(index)]
    for i in range(1, df.shape[0]):
        row_date = df.iloc[i]
        if row_date[0] == str and is_digit(row_date[3]) and is_digit(row_date[4]):
            if not (math.isnan(row_date[1]) and math.isnan(row_date[2])):
                if row_date[2] in [1, 2, 3]:  # 找到对应月份的季度
                    if row_date[1] == 2016:
                        season_data_com['season1-2016'] += row_date[3]
                        season_data_rep['season1-2016'] += row_date[4]
                    else:
                        season_data_com['season1-2017'] += row_date[3]
                        season_data_rep['season1-2017'] += row_date[4]
                elif row_date[2] in [4, 5, 6]:
                    if row_date[1] == 2016:
                        season_data_com['season2-2016'] += row_date[3]
                        season_data_rep['season2-2016'] += row_date[4]
                    else:
                        season_data_com['season2-2017'] += row_date[3]
                        season_data_rep['season2-2017'] += row_date[4]
                elif row_date[2] in [7, 8, 9]:
                    if row_date[1] == 2016:
                        season_data_com['season3-2016'] += row_date[3]
                        season_data_rep['season3-2016'] += row_date[4]
                    else:
                        season_data_com['season3-2017'] += row_date[3]
                        season_data_rep['season3-2017'] += row_date[4]
                elif row_date[2] in [10, 11, 12]:
                    if row_date[1] == 2016:
                        season_data_com['season4-2016'] += row_date[3]
                        season_data_rep['season4-2016'] += row_date[4]
                    else:
                        season_data_com['season4-2017'] += row_date[3]
                        season_data_rep['season4-2017'] += row_date[4]
    x1 = list(season_data_com.keys())
    y1 = list(season_data_com.values())
    x2 = list(season_data_rep.keys())
    y2 = list(season_data_rep.values())
    name1 = '车型对应的季度评论'
    name2 = '车型对应的季度回复'
    addr = '../result/com_rep_man_Plot2.png'
    x_name = '季度'
    y_name = '数量'
    title = '最高评论量车型季度走势'
    graph_plot2(x1, y1, name1, x2, y2, name2, addr, x_name, y_name, title)


def model1_sales_list(df):  # 车型1销量列表
    list_sales_model1 = []
    str = list(dict_model_int.keys())[list(dict_model_int.values()).index(31)]
    for i in range(1, df.shape[0]):
        row_date = df.iloc[i]
        if row_date[2] == str and is_digit(row_date[6]):
            list_sales_model1.append(row_date[6])
    return list_sales_model1


def model1_search_list(df):  # 车型1搜索量列表
    list_sea_model1 = []
    str = list(dict_model_int.keys())[list(dict_model_int.values()).index(31)]
    for i in range(1, df.shape[0]):
        row_date = df.iloc[i]
        if row_date[2] == str and is_digit(row_date[5]):
            list_sea_model1.append(row_date[5])
    return list_sea_model1


def model1_com_list(df):  # 车型1评论量列表
    list_com_model1 = []
    str = list(dict_model_int.keys())[list(dict_model_int.values()).index(31)]
    for i in range(1, df.shape[0]):
        row_date = df.iloc[i]
        if row_date[0] == str and is_digit(row_date[3]):
            list_com_model1.append(row_date[3])
    return list_com_model1


def model1_rep_list(df):  #车型1回复量列表
    list_rep_model1 = []
    str = list(dict_model_int.keys())[list(dict_model_int.values()).index(31)]
    for i in range(1, df.shape[0]):
        row_date = df.iloc[i]
        if row_date[0] == str and is_digit(row_date[4]):
            list_rep_model1.append(row_date[4])
    return list_rep_model1


def graph_bar(x, y, addr, x_name, y_name, title):  # 绘制条形图
    #plt.figure()
    plt.bar(x, y)
    plt.axis('tight')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # plt.xlim((-3, 5))  坐标区间
    plt.title(title)
    #plt.tight_layout(w_pad=3.0)
    plt.savefig(addr)
    plt.show()


def graph_plot(x, y, addr, x_name, y_name, title):  # 绘制折现图
    plt.plot(x, y, marker='.', mec='r', mfc='w')
    plt.axis('tight')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.savefig(addr)
    plt.show()


def graph_plot2(x1, y1, name1, x2, y2, name2, addr, x_name, y_name, title):  # 绘制两条对比折现图
    plt.plot(x1, y1, marker='.', mec='r', mfc='w', label=name1)
    plt.plot(x2, y2, marker='+', ms=10, label=name2)
    plt.legend()
    plt.axis('tight')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.savefig(addr)
    plt.show()


def graph_pie(x, addr, title):  # 绘制饼图
    explode = [0, 0.05, 0, 0.02]
    # explode一个列表，用于指定每块饼片边缘偏离半径的百分比
    plt.pie(list(x), explode=explode,
            labels=list_bodyType, autopct='%3.1f%%',
            startangle=180, shadow=True, colors=['cyan', 'lightpink', 'green', 'yellow'])
    plt.title(title)
    plt.savefig(addr)
    plt.show()


def norm_fun(x, mu, sigma):  # 概率密度函数
    f = np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return f


def graph_f_plot(df, x_l, x_r, x_len, x_name, title, addr):  # 正态分布图
    mean = df.mean()  # 得到均值
    std = df.std()  # 得到标准差
    x = np.arange(x_l, x_r, x_len)
    # 设定 y 轴，载入刚才的正态分布函数
    y = norm_fun(x, mean, std)
    plt.plot(x, y)
    # 画出直方图，最后的“normed”参数，是赋范的意思，数学概念
    plt.hist(df, bins=100, rwidth=0.5, normed=True)
    plt.axis('tight')
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel('Probability')
    plt.savefig(addr)
    plt.show()


def pier(a, b):  # 计算皮尔逊相关系数
    r = stats.pearsonr(a, b)
    return r


# 从表格导入数据
data_sales = pd.read_excel('../data/train_sales_data.xls')  # 读入csv文件
data_search = pd.read_excel('../data/train_search_data.xls')
data_user = pd.read_excel('../data/train_user_reply_data.xls')
data_test = pd.read_excel('../data/test.xls')

add_model(data_sales)  # 获取model数据，并映射

# .shape获取表格行列数
print('数据行列数：')
print("\t销售数据：{}".format(data_sales.shape))
print("\t搜索数据：{}".format(data_search.shape))
print("\t评论数据：{}".format(data_user.shape))
print("\t测试数据：{}".format(data_test.shape))

'''
# 预处理前数据统计图
# #对省份总销量绘图
list_sales_prov = sales_sum_prov(data_sales)
print('各省份的总销量数据：')
print(list_sales_prov)
list_x_sales = list_sales_prov.keys()  # 省份
list_y_sales = list_sales_prov.values()  # 销量
graph_bar(list_x_sales, list_y_sales, '../result/sales_data_Bar.png', '省份', '销量', '各省份总销量条形图')

# #对销量最高和销量最低省份按季度绘图折线图
max_sales = max(list_y_sales)  # 得到最大销量数
province_max_sales = list(list_x_sales)[list(list_y_sales).index(max_sales)]  # 通过value在字典中的下标获取对应的键值
min_sales = min(list_y_sales) # 最小销量数
province_min_sales = list(list_x_sales)[list(list_y_sales).index(min_sales)]
list_sales_max = sales_sum_season(data_sales, province_max_sales)
list_sales_min = sales_sum_season(data_sales, province_min_sales)
print('最高销售量省份分季度销量数据：')
print(list_sales_max)
print('最低销售量省份分季度销量数据：')
print(list_sales_min)
graph_plot(list_sales_max.keys(), list_sales_max.values(), '../result/sales_max_Plot.png', '季度', '销量',
           province_max_sales + '省各季度销量')
graph_plot(list_sales_min.keys(), list_sales_min.values(), '../result/sales_min_Plot.png', '季度', '销量',
           province_min_sales + '省各季度销量')
graph_plot2(list_sales_max.keys(), list_sales_max.values(), province_max_sales,
            list_sales_min.keys(), list_sales_min.values(), province_min_sales,
            '../result/sales_min_Plot.png', '季度', '销量', '最高最低省份季度销量对比')

# #对销售量按车身类型绘图饼图
list_bodyType_sales = body_type_sales_sum(data_sales)
graph_pie(list_bodyType_sales, '../result/bodyType_sales_Pie.png', '车身类型销量图')

# #对销售量求和按车型列表并排序
dict_model_sales = model_sales_sum(data_sales)
print("销售量按车型列表并排序：{}".format(sorted(dict_model_sales.items(), key=lambda x: x[1], reverse=True)))

# #搜索量求和按车型列表并排序
dict_search_sum_model = search_sum_model(data_search)
print("搜索量按车型列表并排序：{}".format(sorted(dict_search_sum_model.items(), key=lambda x: x[1], reverse=True)))
'''
# #评论量求和按车型列表并排序
dict_comment_sum_model = comment_sum_model(data_user)
print("评论量按车型列表并排序：{}".format(sorted(dict_comment_sum_model.items(), key=lambda x: x[1], reverse=True)))

# #某个车型的评论量与回复量对比折线图
max_comment_model = max(dict_comment_sum_model, key=dict_comment_sum_model.get)  # 获取评论量最高的车型
com_rep_sum_season(data_user, max_comment_model)  # 对最高评论量车型绘制评论和回复量季度走势

# 数据清洗部分
# 统计重复记录数
'''
print('数据重复记录行数：')
print("\t销售数据：{:d}".format(data_sales.duplicated().sum()))
print("\t搜索数据：{:d}".format(data_search.duplicated().sum()))
print("\t评论数据：{:d}".format(data_user.duplicated().sum()))
print("\t测试数据：{:d}".format(data_test.duplicated().sum()))

# 删除重复记录行
print('正在删除重复记录数据……')
data_sales = data_sales.drop_duplicates()
data_search = data_search.drop_duplicates()
data_user = data_user.drop_duplicates()
print("删除结束!")

# 删除4个以上缺失值的整行以及空行
print('删除空行以及缺失值有4个及以上的行……')
data_sales.dropna(axis=0, how='all')
data_search.dropna(axis=0, how='all')
data_user.dropna(axis=0, how='all')
data_sales = data_sales.dropna(thresh=4)  # 删除4个以上缺失值的行
data_search = data_search.dropna(thresh=4)
data_user = data_user.dropna(thresh=4)
print('删除结束')
print('新数据行列数：')
print("\t销售数据：{}".format(data_sales.shape))
print("\t搜索数据：{}".format(data_search.shape))
print("\t评论数据：{}".format(data_user.shape))
'''

# 根据已有数据补充部分缺失值以及异常值
# 提取出需要统计的行
'''cat_col = ['regYear', 'province']
d = data_test[cat_col]
c = d['上海']
print(c)
ave_regYear = data_test[data_test['province'].isin('上海')].mean()  # 获取该列的均值
print(ave_regYear)
data_test = data_test.fillna(ave_regYear)  # 用该均值去填充缺失值
print(data_test)'''

# 随机抽样10%数据
# n抽取的行数，frac抽取的比列，replace=True时为有放回抽样，axis=0的时是抽取行，axis=1时是抽取列
sample_sales = data_sales.sample(frac=0.1, replace=True, axis=0)
sample_search = data_search.sample(frac=0.1, replace=True, axis=0)  # frac=0.1
sample_user = data_user.sample(frac=0.1, replace=True, axis=0)

'''
# 求解正态分布、方差、均值、极大极小值
print('样本方差：')
print("\t销售量：{:.2f}".format(sample_sales['salesVolume'].var()))
print("\t搜索量：{:.2f}".format(sample_search['popularity'].var()))
print("\t评论量：{:.2f}".format(sample_user['carCommentVolum'].var()))
print("\t回复量：{:.2f}".format(sample_user['newsReplyVolum'].var()))

print('样本均值：')
print("\t销售量：{:.2f}".format(sample_sales['salesVolume'].mean()))
print("\t搜索量：{:.2f}".format(sample_search['popularity'].mean()))
print("\t评论量：{:.2f}".format(sample_user['carCommentVolum'].mean()))
print("\t回复量：{:.2f}".format(sample_user['newsReplyVolum'].mean()))

print('样本极值：')
print("\t销售量极大值：{0}, 极小值：{1}".format(data_sales['salesVolume'].max(), data_sales['salesVolume'].min()))
print("\t搜索量极大值：{0}, 极小值：{1}".format(data_search['popularity'].max(), data_search['popularity'].min()))
print("\t评论量极大值：{0}, 极小值：{1}".format(data_user['carCommentVolum'].max(), data_user['carCommentVolum'].min()))
print("\t回复量极大值：{0}, 极小值：{1}".format(data_user['newsReplyVolum'].max(), data_user['newsReplyVolum'].min()))'''

# #画正态分布图
'''graph_f_plot(data_sales['salesVolume'], -1000, 3000, 0.1, '销售量', '销售量正态分布图', '../result/data_sales_f.png')
graph_f_plot(data_search['popularity'], -5000, 20000, 1, '搜索量', '搜索量正态分布图', '../result/data_search_f.png')'''

'''
# 数据相关性，另确定一定的事故发生率导致的评论数上升，即(1 - 评论与销量相关性)/2
# #验证搜索量与销量的相关性，必须为同一车型，这里取车型31，抽取500个数据
data_sea_sales = pd.DataFrame({'搜索量': model1_search_list(data_search), '销售量': model1_sales_list(data_sales)})
sample_sea_sales = data_sea_sales.sample(n=500, replace=False, axis=0)
search_a1 = sample_sea_sales['搜索量']
sales_a1 = sample_sea_sales['销售量']
sea_rel_sales_rate = pier(sales_a1, search_a1)
print("搜索量与销售量的相关性：{}".format(sea_rel_sales_rate))

# #验证评论与回复量的相关性
data_com_rep = pd.DataFrame({'评论量': model1_com_list(data_user), '回复量': model1_rep_list(data_user)})
sample_com_rep = data_com_rep.sample(n=20, replace=False, axis=0)
com_c1 = sample_com_rep['评论量']
rep_c1 = sample_com_rep['回复量']
com_rel_rep_rate = pier(com_c1, rep_c1)
print("评论量与回复量的相关性：{}".format(com_rel_rep_rate))

accidence_rel_rate = (1 - sea_rel_sales_rate[0]) / 2
print("事故导致搜索量增加率：{:.4f}".format(accidence_rel_rate))
'''
