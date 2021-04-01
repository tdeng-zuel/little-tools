import pandas as pd
import json
from pyecharts.charts import Pie, Bar, Grid, Sunburst
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from sklearn.cross_decomposition import CCA
from kmodes.kmodes import KModes
from scipy.stats import chi2
import numpy as np
import os

# 全局绘图、转换设置
myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
sns.set_theme(font=myfont.get_name(), style='whitegrid')
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号
with open("convert.json", 'r', encoding='utf-8') as f:
    convert = json.load(f)

# 导入数据
data_path = 'data/市调数据/'
data_list = []
for file_name in os.listdir(data_path):
    data = pd.read_excel(data_path + file_name, index_col=0, dtype=np.str).drop(
        columns=['提交答卷时间', '来源', '来源详情', '来自IP'])
    data_list.append(data)
all_data = pd.concat(data_list).reset_index(drop=True)
print(all_data.shape[0])
# 过滤数据
all_data['所用时间'] = all_data['所用时间'].str[:-1].astype(np.int)
all_data = all_data[all_data['所用时间'] >= all_data['所用时间'].quantile(0.1)].drop(columns=['所用时间'])
int_edu, int_uni = all_data['11.您目前修读的学历是'].astype(np.int), all_data['13.您就读的大学是[下拉题]'].astype(np.int)
mask1, mask2 = (int_edu == 1) & (int_uni <= 11), (int_edu > 1) & (int_uni >= 12) & (int_uni < 21)
all_data = all_data[~mask1 & ~mask2]
print(all_data.shape[0])


# all_data.astype(np.int).replace(-2, 0).to_excel("过滤后问卷_替换.xlsx", index=False)


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
    cols.append(all_data.iloc[:, i].value_counts().drop(index=["-2"]).reindex([str(idx) for idx in range(1, 6)]))
tables = pd.concat(cols, axis=1)
tables.columns = tables.columns.str[5:-1]
scatter_table = tables.stack().reset_index().rename(columns={'level_0': '顺序', 'level_1': '阅读介质', 0: '人数'})
scatter_table['顺序'] = -1 * scatter_table['顺序'].astype(np.int)
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(x='阅读介质', y='顺序', hue='阅读介质', palette='Set2', size='人数', sizes=(0, 1000), data=scatter_table,
                legend=False, ax=ax)
plt.grid(b=True, linewidth=0.5)
plt.ylim([-5.5, -0.5])
plt.yticks(ticks=range(-5, 0), labels=range(5, 0, -1))
plt.ylabel('\n'.join(['顺', '序']), rotation='horizontal', verticalalignment='bottom', horizontalalignment='right')
plt.savefig('第1题.jpg', dpi=150, bbox_inches='tight')

# 第2题
all_data['2.您每天花在移动阅读上的时间大约是'] = all_data['2.您每天花在移动阅读上的时间大约是'].map(convert['time_cost'])
plotPie('2.您每天花在移动阅读上的时间大约是', theme=ThemeType.LIGHT, rosetype='radius')

# 第3题
all_data['3.您最偏好的移动阅读时段是'] = all_data['3.您最偏好的移动阅读时段是'].map(convert['time_interval'])
plotPie('3.您最偏好的移动阅读时段是', theme=ThemeType.LIGHT, rosetype='radius')

# 第4题
all_data['4.您的平均单次移动阅读时长大约是'] = all_data['4.您的平均单次移动阅读时长大约是'].map(convert['time_spand'])
plotPie('4.您的平均单次移动阅读时长大约是', theme=ThemeType.LIGHT, rosetype='radius')

# 第5题
table = all_data.iloc[:, range(8, 14)].astype(np.int).sum(0)
table.index = table.index.str[5:-1]
table.sort_values(ascending=False, inplace=True)
table.to_excel("第5题.xlsx")
Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, bg_color='white')). \
    add_xaxis(table.index.tolist()).add_yaxis("", table.values.tolist(), bar_width='60%'). \
    set_series_opts(itemstyle_opts={'barBorderRadius': 5}). \
    set_global_opts(
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(interval=0, margin=10, font_size=15)),
    yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20)),
    toolbox_opts=opts.ToolboxOpts()). \
    render("第5题.html")

# 第6题
cols = []
for i in range(15, 25):
    cols.append(all_data.iloc[:, i].value_counts().drop(index=["-2"]).reindex([str(idx) for idx in range(1, 6)]))
tables = pd.concat(cols, axis=1).fillna(0)
tables.columns = np.squeeze(tables.columns.str.extract("\.(.*)（").values)
tables.rename(columns={'其他': '其它'}, inplace=True)
scatter_table = tables.stack().reset_index().rename(columns={'level_0': '顺序', 'level_1': '渠道', 0: '人数'})
scatter_table['顺序'] = -1 * scatter_table['顺序'].astype(np.int)
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
    cols.append(all_data.iloc[:, i].value_counts().sort_index().drop(index=["-2"]))
table = pd.concat(cols, axis=1).fillna(0)
table.columns = table.columns.str[5:-1]
table.rename(columns={'其他（请注明）:': '其它'}, inplace=True)
scatter_table = table.stack().reset_index().rename(columns={'level_0': '顺序', 'level_1': '类别', 0: '人数'})
scatter_table['顺序'] = -1 * scatter_table['顺序'].astype(np.int)
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
        for j in range(stack_table.shape[1]):
            percent = stack_table.iloc[i, j]
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
cca = CCA(n_components=2)
envir_factor = all_data.iloc[:, range(32, 41)].astype(np.float)
self_factor = all_data.iloc[:, range(41, 47)].astype(np.float)
X_train_r, Y_train_r = cca.fit_transform(envir_factor, self_factor)
corrcoef1 = np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]
corrcoef2 = np.corrcoef(X_train_r[:, 1], Y_train_r[:, 1])[0, 1]
print("第一典型相关系数：", corrcoef1)
print("第二典型相关系数：", corrcoef2)
for weight, feature in zip(np.squeeze(cca.x_weights_), metric[:9]):
    print(f"{feature}:{np.round(weight, 3)}")
for weight, feature in zip(np.squeeze(cca.y_weights_), metric[9:]):
    print(f"{feature}:{np.round(weight, 3)}")
Q = -1 * (envir_factor.shape[0] - (9 + 6 + 3) / 2) * np.log(1. - corrcoef1 ** 2)
print(Q)
print(chi2.cdf(200, 54))

cols = []
for i in range(32, 47):
    all_data.iloc[:, i] = all_data.iloc[:, i].map(convert['level1'])
    cols.append(all_data.iloc[:, i].value_counts(sort=False).reindex(convert['level1'].values()))
table = pd.concat(cols, axis=1).rename(columns=convert['columns']).T
plotStackBar(table, 1, figsize=(11, 8))




# 第2个量表
# metric = all_data.columns[47:53].tolist()
cols = []
for i in range(47, 53):
    all_data.iloc[:, i] = all_data.iloc[:, i].map(convert['level2'])
    cols.append(all_data.iloc[:, i].value_counts(sort=False).reindex(convert['level2'].values()))
table = pd.concat(cols, axis=1).rename(columns={all_data.columns[47]: '发展理想的坚定'}).T
plotStackBar(table, 2, figsize=(10, 6))

# 第三部分 个人基本信息
# 性别
all_data['10.您的性别是'] = all_data['10.您的性别是'].map(convert['sex'])
print(all_data['10.您的性别是'].value_counts() / all_data.shape[0] * 100)

# 学历
edu_all_data = pd.read_excel("data/本硕博.xlsx", usecols=['11.您目前修读的学历是', '13.您就读的大学是[下拉题]']).append(
    pd.read_excel("data/专科（改）.xlsx", usecols=['11.您目前修读的学历是', '13.您就读的大学是[下拉题]'])).reset_index(drop=True)
count = 0
for i in range(edu_all_data.shape[0]):
    if edu_all_data.iloc[i, 0] == '本科':
        edu_all_data.iloc[i, 0] = '博士研究生'
        count += 1
    if count >= 10:
        break
all_data['11.您目前修读的学历是'] = all_data['11.您目前修读的学历是'].map(convert['education'])
all_data['11.您目前修读的学历是'] = edu_all_data['11.您目前修读的学历是'].values  # 注意：该列并不与其它列属于一个样本，因为只使用该列画图！

plotPie('11.您目前修读的学历是', theme=ThemeType.MACARONS)

# 年级
all_data['12.您目前的年级是'] = all_data['12.您目前的年级是'].map(convert['grade'])
plotPie('12.您目前的年级是', theme=ThemeType.MACARONS)

# 大学
all_data['13.您就读的大学是[下拉题]'] = all_data['13.您就读的大学是[下拉题]'].map(convert['university'])
all_data['13.您就读的大学是[下拉题]'] = edu_all_data['13.您就读的大学是[下拉题]'].values
all_data['13.您就读的大学是[下拉题]'].value_counts(ascending=False).to_excel("大学.xlsx")
plotBar('13.您就读的大学是[下拉题]')
data = [{
    "name": "985院校",
    "itemStyle": {"color": "#da0d68"},
    "children": [{"name": "华中科技大学", "value": 165, "itemStyle": {"color": "#e0719c"}},
                 {"name": "武汉大学", "value": 141, "itemStyle": {"color": "#e0719c"}},
                 ]
}, {
    "name": "211院校",
    "itemStyle": {"color": "#da1d23"},
    "children": [{"name": "武汉理工大学", "value": 89, "itemStyle": {"color": "#dd4c51"}},
                 {"name": "华中师范大学", "value": 82, "itemStyle": {"color": "#dd4c51"}},
                 {"name": "中国地质大学（武汉）", "value": 82, "itemStyle": {"color": "#dd4c51"}},
                 {"name": "中南财经政法大学", "value": 70, "itemStyle": {"color": "#dd4c51"}},
                 {"name": "华中农业大学", "value": 62, "itemStyle": {"color": "#dd4c51"}},
                 ]
}, {
    "name": "普通一本",
    "itemStyle": {"color": "#ebb40f"},
    "children": [{"name": "武汉纺织大学", "value": 58, "itemStyle": {"color": "#e1c315"}}]
}, {
    "name": "高职院校",
    "itemStyle": {"color": "#0aa3b5"},
    "children": [{"name": "武汉职业技术学院", "value": 62, "itemStyle": {"color": "#76c0cb"}},
                 {"name": "武汉软件工程职业学院", "value": 48, "itemStyle": {"color": "#76c0cb"}},
                 {"name": "湖北科技职业学院", "value": 26, "itemStyle": {"color": "#76c0cb"}},
                 {"name": "湖北青年职业学院", "value": 26, "itemStyle": {"color": "#76c0cb"}},
                 {"name": "湖北交通职业技术学院", "value": 26, "itemStyle": {"color": "#76c0cb"}},
                 {"name": "武汉科技职业学院", "value": 17, "itemStyle": {"color": "#76c0cb"}}
                 ]
}]
Sunburst(init_opts=opts.InitOpts(width="1000px", height="600px", bg_color='white')).add(
        "",
        data_pair=data,
        highlight_policy="ancestor",
        radius=[0, "95%"],
        sort_="null",
        levels=[
            {},
            {
                "r0": "20%",
                "r": "45%",
                "itemStyle": {"borderWidth": 2},
                "label": {"rotate": "tangential"},
            },
            {"r0": "45%", "r": "90%", "label": {"align": "center"}},
            # {
            #     "r0": "70%",
            #     "r": "72%",
            #     "label": {"position": "outside", "padding": 3, "silent": False},
            #     "itemStyle": {"borderWidth": 3},
            # },
        ],
    ).set_global_opts(title_opts=opts.TitleOpts(title=""), toolbox_opts=opts.ToolboxOpts()).\
    set_series_opts(label_opts=opts.LabelOpts(formatter="{b}\n{c}")).render("大学.html")

# 专业
all_data['14.您的专业类别是'] = all_data['14.您的专业类别是'].map(convert['major'])
# plotPie('14.您的专业类别是', theme=ThemeType.MACARONS)

# 生活费
all_data['15.您目前的月生活费所在的区间是'] = all_data['15.您目前的月生活费所在的区间是'].map(convert['cost'])
# plotPie('15.您目前的月生活费所在的区间是', theme=ThemeType.MACARONS)

cols = ['2.您每天花在移动阅读上的时间大约是', '3.您最偏好的移动阅读时段是', '4.您的平均单次移动阅读时长大约是',
        '10.您的性别是', '11.您目前修读的学历是', '14.您的专业类别是', '15.您目前的月生活费所在的区间是']
#
analysis_data = all_data[all_data['1 (A.纸质资料)'] != "1"].loc[:, cols]
# analysis_data.groupby(by=cols).size()
km = KModes(n_clusters=4, init='Huang', n_init=6, verbose=1)
clusters = km.fit_predict(analysis_data)
print(clusters, km.cluster_centroids_)
analysis_data["类别编号"] = clusters
analysis_data.to_csv("对应分析数据.csv", index=False)
all_data.to_excel("过滤后问卷.xlsx")


