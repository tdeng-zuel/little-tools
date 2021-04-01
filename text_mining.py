import pandas as pd
import numpy as np
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import jieba

# 导入数据
text = pd.read_table("data/中南野史演义.txt", header=None)
# 过滤推文标题
text = text[~text.loc[:, 0].str.startswith("中南野史演义(") & ~text.loc[:, 0].str.startswith("Evan")].reset_index(drop=True)
# 过滤长度为1的词
my_words = " ".join([word for word in jieba.lcut("".join([text.iloc[i, 0] for i in range(text.shape[0])])) if len(word) != 1])
# 构建词云
wordcloud = WordCloud(width=1400, height=1000, font_step=50, font_path="data/FZZJ-WHBZTJW.TTF", background_color='white',
                      mask=np.array(Image.open("data/DJ.webp")), max_font_size=150).generate(my_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
wordcloud.to_file("wordcloud-cn.jpg")

