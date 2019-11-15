# 数据预处理-车辆消费数据-设计文档

数据源：
60个车型在22个省份，从2016年1月至2017年12月的销量。训练集：
历史销量train_sales_data.csv；车型搜索：train_search_data.csv；汽车垂直媒体新闻评论和车型评论数据：train_user_reply_data.csv.

数据特性：
在时间上，按照时间排序，且在时间上是连续的，省份和省份编码是一一对应的，车型和车身的取值是固定的

一、实现任务：
1.数据统计描述，绘制条形图、折线图、饼图，输出均值、方差、极值、总和等
2.数据清洗，空值、异常值、非合理值处理并填充合理的值
3.简单数据变换、映射
4.根据数据相关度预测未来1个月某个省份的销量


二、采用的方法和算法
1. 使用了哪些方法、算法、工具?
    (1).对数据的抽象统计有对省份-销量、时间-销量、车型-评论量等绘制条形图、折线图以及饼图。
    在预处理前和与处理后分别抽样统计，得到结果对比图
    (2).在数据清洗中，删除重复数据行和空数据行；首先查看缺失值，即值为空的属性值；
    作相应的数据取舍，取舍规则是：
        (a).如果该行数据缺失4个及以上的属性值删除该行
        (b).如果该行缺失province(省份)，则用相应adcode(省份编码)对应的省份名称替换，如果两项均缺失则删除该行
        (c).如果该行缺失adcode(省份编码)，则用对应的省份名称替换，如果两项均缺失则删除该行
        (d).如果该行缺失model(车型编码)，则用该省份model的众数替换
        (e).如果该行缺失bodyType(车身类型)，则用该省份bodyType的众数替换
        (f).如果该行缺失regYear(年份)，则用上下文年份代替，表1中用上文regYear和regMonth；
        表2根据regMonth的递增序(同一地区在一起、按时间排列)；
        表3和表2一样(按照model分在一起、按时间排序)
        (g).如果该行缺失regMonth同(f)
        (h).如果该行缺失salesVolume(销量)，则用该省份的salesVolume均值替换
        (i).如果该行缺失popularity(搜索量)，则用该省份前后5个popularity的均值替换
        (j).如果该行缺失carComment(车型评价量)，附近两个的均值
        (k).如果该行缺失newsReply(新闻评论量)同(j)
    判断各列属性值是否满足相应的数据类型，即是否合法(不考虑合理性)，如果不合法则根据取舍规则替换；
    再判断数据异常值，即数据合理性，极大或极小值(数据不用正态分布清除)的数据，如果出现异常则根据取舍规则替换。
    (3).数据简单变换或映射
    采用了字典和列表的数据结构，将字符串等类型的长数据转化为简单地易于分辨的数据
    例如：
        list_prov_code = {  # <province, adcode>
            '河南': '410000', '广东': '440000', '山东': '370000', '河北': '130000', '江苏': '320000', '四川': '510000',
            '上海': '310000', '云南': '530000', '陕西': '610000', '山西': '140000', '广西': '450000', '重庆': '150000',
            '内蒙古': '320000', '湖南': '430000', '北京': '110000', '安徽': '340000','辽宁': '210000',
            '黑龙江': '230000', '江西': '360000', '福建': '350000', '浙江': '330000', '湖北': '420000'
        }
        dict_model_int.update({model: i})  # <model, int>
        <model, sales> = [
            (17, 918371), (5, 809207), (14, 797813), (10, 767919), (38, 764605), (6, 637974),
            (21, 636136), (2, 629960), (16, 616402), (32, 608190), (18, 519295), (11, 431510),
            (56, 418193), (9, 404738), (22, 401644), (13, 387117), (20, 361915), (19, 349160),
            (39, 339375), (15, 338746), (53, 325081), (4, 315615), (36, 309220), (42, 305862),
            (40, 304141), (43, 296490), (59, 290718), (12, 288383), (47, 285154), (49, 284053),
            (55, 280368), (33, 274178), (30, 243906), (1, 234414), (24, 228508), (48, 223416),
            (31, 219778), (57, 215209), (7, 213237), (51, 212379), (28, 207880), (52, 203644),
            (35, 197287), (50, 184754), (26, 183422), (3, 175322), (58, 171576), (29, 169356),
            (45, 161215), (37, 139110), (8, 135446), (23, 135327), (25, 130685), (46, 108883),
            (54, 100004), (60, 90903), (41, 82050), (34, 77771), (44, 75115), (27, 65483)
        ]
    (4).数据相关性，以及预测未来趋势
        (a).对表格数据随机抽样，验证同一车型销售量和评论量、评论量和回复量
        (b).皮尔森相关系数，适用于线性函数的自变量和因变量，协方差/两变量方差积
        第二个值为p-value，统计学上，一般当p-value < 0.05时，可以认为两变量存在相关性。
        (c).搜索量在一定程度上会增加销量，但偶尔的事故导致搜索量上升，但销量反而降低，考虑事故发生率
    (5).采用的库有pandas、numpy、matplotlib、missingno等，通过库函数获取相应的操作和数据；
    通过制作统计图来可视化表示数据分布；scipy是皮尔逊相关系数的包

2. 为何采用这些方法?
    (1).首先确保数据的规范性和合理性，考虑到缺失的数据也是重要的数据参考，所以应该采取合理地填补或者替换，用均值、众数等替换
	(2).然后对数据进行描述统计，可以预测未来销量趋势
	(3).求出数据之间的相关度，看数据之间的影响度
	(4).数据变换和映射，将大数据或难以观看的数据简化，方便处理数据

3. 请描述具体的数据分析步骤。
	(1).查看表格数据基本信息：行列数、重复记录数
	(2).数据清洗：主要是删除重复数据并填补，处理异常数据
	(3).求出样本均值、方差、极值等
	(4).绘制数据条形图、折线图、饼图等
	(5).求出数据之间的相关性、分析数据之间的影响
	(6).得出预测结论

三、结果和规律
1. 采用上述方法的分析结果如何?
    预处理后的统计效果图：


	效果就是解决了一些过大的误差或者错误导致的预测不精准，还有就是对多种数据相关造成的影响进行分析

2. 通过上述分析，对于数据集的性质/特征/分布有何发现?
	(1).在时间上，按照时间排序，且在时间上是连续的，省份和省份编码是一一对应的，车型和车身的取值是固定的；
	(2).评论量与回复量对应着同一个车型，但在处理两个表格数据相关性的时候，需要保证数据信息一致；
	(3).回复量随着评论量的增加而增加，在2016年初最少，之后起起伏伏地增加；
	(4).搜索量与销量相关性最强，且呈正相关。

3. 当前分析结果将怎样辅助用户应对最终的分析需求(大赛数据可参考竞赛题说明)?


本部分建议上传不超过三张图片和一组数据结果，以说明完成情况。数据结果如果大于10M，请使用外部链接。
