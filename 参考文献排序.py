import re
from pypinyin import lazy_pinyin
'''说明：分中英文文献按照字母顺序升序排列。乱序的参考文献放在myarticle.txt中，格式为
    [序号] 参考文献.年份这样，生成的参考文献放在new_article.txt'''

new_file = open('new_article.txt', 'w')
pattern1 = r'\s\D+?[．，]'
pattern2 = r'\[[0-9]*\]'
pattern3 = r'[a-zA-Z]'
pattern4 = r'[12]\d{3}'
regex1 = re.compile(pattern1, flags=re.IGNORECASE)
regex2 = re.compile(pattern2, flags=re.IGNORECASE)
regex3 = re.compile(pattern3, flags=re.IGNORECASE)
regex4 = re.compile(pattern4, flags=re.IGNORECASE)
writer_article_list = []
chinese_tuple = []
eng_tuple = []
chinese_list = []
eng_list = []
test = 'efijn200034lskrf'

with open('myarticle.txt', 'r', encoding='utf-8') as file:
    for line in file.readlines():
        writer_article_list.append((line[regex1.search(line).start() + 1:regex1.search(line).end() - 1],
                                    line, line[regex4.search(line).start():regex4.search(line).end()]))

    for key, value, year in writer_article_list:
        if regex3.search(key):
            eng_tuple.append((key, value, year))
        else:
            chinese_tuple.append((''.join(lazy_pinyin(key)), value, year))

    for _ in sorted(chinese_tuple, key=lambda x: (x[0], x[2])):
        chinese_list.append(_[1])

    for _ in sorted(eng_tuple, key=lambda x: (x[0], x[2])):
        eng_list.append(_[1])
    for i in range(len(chinese_list)):
        text = chinese_list[i]
        chinese_list[i] = text.replace(text[regex2.search(text).start() + 1:regex2.search(text).end() - 1], str(i + 1), 1)
    print(chinese_list)
    for j in range(len(eng_list)):
        text = eng_list[j]
        eng_list[j] = text.replace(text[regex2.search(text).start() + 1:regex2.search(text).end() - 1],
                                   str(j + 1 + len(chinese_list)), 1)
    final_list = chinese_list + eng_list
    for line in final_list:
        new_file.write(line)
