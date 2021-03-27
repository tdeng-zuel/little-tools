import pandas as pd
from pyecharts.charts import Pie, Bar, Grid
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import numpy as np
import os

myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
sns.set_theme(font=myfont.get_name(), style='whitegrid')
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 导入数据
data_path = 'data/市调数据/'
data_list = []
for file_name in os.listdir(data_path):
    data = pd.read_excel(data_path + file_name, index_col=0).drop(columns=['提交答卷时间', '来源', '来源详情', '来自IP'])
    data_list.append(data)
all_data = pd.concat(data_list).reset_index(drop=True)
all_data['所用时间'] = all_data['所用时间'].str[:-1].astype(np.int)
print(all_data[all_data['所用时间'] < all_data['所用时间'].quantile(0.1)])


def plotPie(col_name, theme, rosetype=None):
    global all_data
    file_name = col_name.split('的')[1][:-1]
    counts = all_data[col_name].value_counts()
    Pie(init_opts=opts.InitOpts(theme=theme, width='1050px', height='750px', bg_color='white')).add(
        "", [list(z) for z in zip(counts.index.tolist(), counts.values.tolist())],
        radius=["40%", "75%"], rosetype=rosetype
    ).set_global_opts(
        legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%", is_show=False),
        toolbox_opts=opts.ToolboxOpts()

    ).set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%", font_size=20)).render(file_name + '.html')


def plotBar(col_name):
    global all_data
    file_name = col_name.split('的')[1] + '.html'
    counts = all_data[col_name].value_counts(ascending=True)
    grid = Grid(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='900px', height='700px', bg_color='white'))
    bar = Bar().add_xaxis(counts.index.tolist()). \
        add_yaxis("", counts.values.tolist()).reversal_axis().set_series_opts(
        itemstyle_opts={'barBorderRadius': 5},
        label_opts=opts.LabelOpts(position="right", font_size=15)).set_global_opts(
        toolbox_opts=opts.ToolboxOpts(),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=15)),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20)))

    grid.add(bar, grid_opts=opts.GridOpts(pos_left="25%")).render(file_name)


# 第一部分 移动阅读基本情况
# 第1题
cols = []
for i in range(0, 5):
    cols.append(all_data.iloc[:, i].value_counts().drop(index=[-2]).reindex(range(1, 6)))
tables = pd.concat(cols, axis=1)
tables.columns = tables.columns.str[5:-1]
scatter_table = tables.stack().reset_index().rename(columns={'level_0': '顺序', 'level_1': '阅读介质', 0: '人数'})
scatter_table['顺序'] *= -1
fig = plt.figure(figsize=(9, 6))
sns.scatterplot(x='阅读介质', y='顺序', hue='阅读介质', palette='Set2', size='人数', sizes=(0, 1000), data=scatter_table,
                legend=False)
plt.grid(b=True, linewidth=0.5)
plt.ylim([-5.5, -0.5])
plt.yticks(ticks=range(-5, 0), labels=range(5, 0, -1))
plt.ylabel('\n'.join(['顺', '序']), rotation='horizontal', verticalalignment='bottom', horizontalalignment='right')
plt.savefig('第1题.jpg', dpi=150, bbox_inches='tight')

# 第2题
time_cost = {1: '2小时以下', 2: '2-4小时', 3: '4-6小时', 4: '6-8小时', 5: '8小时以上'}
all_data['2.您每天花在移动阅读上的时间大约是'] = all_data['2.您每天花在移动阅读上的时间大约是'].map(time_cost)
plotPie('2.您每天花在移动阅读上的时间大约是', theme=ThemeType.LIGHT, rosetype='radius')

# 第3题
time_interval = {1: '0-9点', 2: '9-12点', 3: '12-15点', 4: '15-18点', 5: '18-24点'}
all_data['3.您最偏好的移动阅读时段是'] = all_data['3.您最偏好的移动阅读时段是'].map(time_interval)
plotPie('3.您最偏好的移动阅读时段是', theme=ThemeType.LIGHT, rosetype='radius')

# 第4题
time_spand = {1: '10分钟以内', 2: '10-30分钟', 3: '30-60分钟', 4: '60-90分钟', 5: '90分钟以上'}
all_data['4.您的平均单次移动阅读时长大约是'] = all_data['4.您的平均单次移动阅读时长大约是'].map(time_spand)
plotPie('4.您的平均单次移动阅读时长大约是', theme=ThemeType.LIGHT, rosetype='radius')

# 第5题
table = all_data.iloc[:, range(8, 14)].sum(0)
table.index = table.index.str[5:-1]
table.sort_values(ascending=False, inplace=True)
Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT)). \
    add_xaxis(table.index.tolist()).add_yaxis("", table.values.tolist(), bar_width='60%'). \
    set_series_opts(itemstyle_opts={'barBorderRadius': 5}). \
    set_global_opts(
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(interval=0, margin=10)),
    toolbox_opts=opts.ToolboxOpts()). \
    render("第5题.html")

# 第6题
cols = []
for i in range(15, 25):
    cols.append(all_data.iloc[:, i].value_counts().drop(index=[-2]).reindex(range(1, 6)))
tables = pd.concat(cols, axis=1).fillna(0)
tables.columns = np.squeeze(tables.columns.str.extract("\.(.*)（").values)
tables.rename(columns={'其他': '其它'}, inplace=True)
scatter_table = tables.stack().reset_index().rename(columns={'level_0': '顺序', 'level_1': '渠道', 0: '人数'})
scatter_table['顺序'] *= -1
fig = plt.figure(figsize=(11, 6))
sns.scatterplot(x='渠道', y='顺序', hue='渠道', palette='Paired', size='人数', sizes=(0, 1000), data=scatter_table,
                legend=False)
plt.grid(b=True, linewidth=0.5)
plt.ylim([-5.5, -0.5])
plt.yticks(ticks=range(-5, 0), labels=range(5, 0, -1))
plt.ylabel('\n'.join(['顺', '序']), rotation='horizontal', verticalalignment='bottom', horizontalalignment='right')
plt.savefig('第6题.jpg', dpi=150, bbox_inches='tight')

# 第7题
cols = []
for i in range(25, 32):
    cols.append(all_data.iloc[:, i].value_counts().sort_index().drop(index=[-2]))
table = pd.concat(cols, axis=1).fillna(0)
table.columns = table.columns.str[5:-1]
table.rename(columns={'其他（请注明）:': '其它'}, inplace=True)
scatter_table = table.stack().reset_index().rename(columns={'level_0': '顺序', 'level_1': '类别', 0: '人数'})
scatter_table['顺序'] *= -1
fig = plt.figure(figsize=(9, 6))
sns.scatterplot(x='类别', y='顺序', hue='类别', palette='Set2', size='人数', sizes=(0, 1000), data=scatter_table, legend=False)
plt.grid(b=True, linewidth=0.5)
plt.ylim([-7.5, -0.5])
plt.yticks(ticks=range(-7, 0), labels=range(7, 0, -1))
plt.ylabel('\n'.join(['顺', '序']), rotation='horizontal', verticalalignment='bottom', horizontalalignment='right')
plt.savefig('第7题.jpg', dpi=150, bbox_inches='tight')


# 第二部分 移动阅读体验评价
def plotStackBar(table, table_index, figsize):
    stack_table = np.round(table.div(np.sum(table, axis=1), axis=0) * 100, 2)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    stack_table.reindex(stack_table.index[::-1]).plot(
        kind='barh', stacked=True, ax=ax, colormap='Set3', width=0.7, xlim=[0, 100], fontsize=15, zorder=1)
    plt.axvline(x=50, color='red', zorder=0)
    plt.xticks(ticks=range(0, 101, 10), labels=[str(percent) + '%' for percent in range(0, 101, 10)])
    plt.legend(bbox_to_anchor=(0.5, -0.07), loc="upper center", borderaxespad=0., ncol=5)
    for i in range(stack_table.shape[0]):
        ave = 0
        for j in range(stack_table.shape[1]):
            percent = stack_table.iloc[i, j]
            ave += (j + 1) * percent
            if percent > 5:
                plt.text(stack_table.iloc[i, :j].sum() + percent / 2,
                         stack_table.shape[0] - i - 1, str(percent) + '%',
                         va='center', ha='center', size=13)
    for lb in ax.xaxis.get_ticklabels():
        if lb.get_text() == '50%':
            lb.set_color('red')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(table_index).join(['量表', '.jpg']), dpi=120, bbox_inches='tight')


# 第1个量表
metric = all_data.columns[33:(32 + 15)].tolist()
metric.insert(0, all_data.columns[32].split('—')[1])
level = {1: '非常低', 2: '较低', 3: '一般', 4: '较高', 5: '非常高'}
cols = []
for i in range(32, 47):
    all_data.iloc[:, i] = all_data.iloc[:, i].map(level)
    cols.append(all_data.iloc[:, i].value_counts(sort=False).reindex(level.values()))
table = pd.concat(cols, axis=1).rename(
    columns={all_data.columns[32]: '载体便携性', '移动阅读材料的即时性': '材料即时性',
             '移动阅读资源的充足性': '资源充足性', '移动阅读界面的友好性': '界面友好性',
             '移动阅读功能的多样性': '功能多样性', '移动阅读内容的质量': '移动阅读内容质量',
             '教育制度对移动阅读的包纳程度': '教育制度包纳程度', '周边氛围对移动阅读的欢迎程度': '周边氛围欢迎程度',
             '校园设施对移动阅读的支持程度': '校园设施支持程度', '自身在移动阅读中的专注度': '自身的专注程度',
             '自身在移动阅读中的耐心': '自身的耐心程度', '自身在移动阅读中的思考深度': '自身的思考深度',
             '自身在移动阅读中的涉猎广度': '自身的涉猎广度', '自身对移动阅读资源的筛选能力': '自身的筛选能力',
             '自身对移动阅读内容的应用能力': '自身的应用能力'}).T
plotStackBar(table, 1, figsize=(11, 8))


# 第2个量表
metric = all_data.columns[47:53].tolist()
level = {1: '非常抑制', 2: '比较抑制', 3: '无明显影响', 4: '比较促进', 5: '非常促进'}
cols = []
for i in range(47, 53):
    all_data.iloc[:, i] = all_data.iloc[:, i].map(level)
    cols.append(all_data.iloc[:, i].value_counts(sort=False).reindex(level.values()))
table = pd.concat(cols, axis=1).rename(columns={all_data.columns[47]: '发展理想的坚定'}).T
plotStackBar(table, 2, figsize=(10, 6))

# 第三部分 个人基本信息
# 性别
sex = {1: '男', 2: '女'}
all_data['10.您的性别是'] = all_data['10.您的性别是'].map(sex)
print(all_data['10.您的性别是'].value_counts() / all_data.shape[0] * 100)

# 学历
education = {1: '专科', 2: '本科', 3: '硕士研究生', 4: '博士研究生'}
all_data['11.您目前修读的学历是'] = all_data['11.您目前修读的学历是'].map(education)
plotPie('11.您目前修读的学历是', theme=ThemeType.MACARONS)

# 年级
grade = {1: '一年级', 2: '二年级', 3: '三年级', 4: '四年级', 5: '五年级及以上'}
all_data['12.您目前的年级是'] = all_data['12.您目前的年级是'].map(grade)
plotPie('12.您目前的年级是', theme=ThemeType.MACARONS)

# 大学
university = {1: '武汉大学', 2: '华中科技大学', 3: '中南财经政法大学', 4: '中国地质大学（武汉）', 5: '华中农业大学',
              6: '华中师范大学', 7: '武汉理工大学', 8: '湖北工业大学', 9: '武汉纺织大学', 10: '中南民族大学', 11: '江汉大学',
              12: '武汉职业技术学院', 13: '长江职业学院', 14: '湖北交通职业技术学院', 15: '武汉软件工程职业学院', 16: '湖北开放职业学院',
              17: '武汉科技职业学院', 18: '武汉民政职业学院', 19: '湖北科技职业学院', 20: '湖北青年职业学院', 21: '其它'}
all_data['13.您就读的大学是[下拉题]'] = all_data['13.您就读的大学是[下拉题]'].map(university)
plotBar('13.您就读的大学是[下拉题]')

# 专业
major = {1: '哲学', 2: '经济学', 3: '法学', 4: '教育学', 5: '文学', 6: '历史学', 7: '理学', 8: '工学', 9: '农学', 10: '医学',
         11: '管理学', 12: '艺术学'}
all_data['14.您的专业类别是'] = all_data['14.您的专业类别是'].map(major)
plotPie('14.您的专业类别是', theme=ThemeType.MACARONS)

# 生活费
cost = {1: '1000元以下', 2: '1000-1500元', 3: '1500-2000元', 4: '2000-3000元', 5: '3000元以上'}
all_data['15.您目前的月生活费所在的区间是'] = all_data['15.您目前的月生活费所在的区间是'].map(cost)
plotPie('15.您目前的月生活费所在的区间是', theme=ThemeType.MACARONS)
