---
layout: post
category: "other"
title:  "爬虫抓取微博内容"
tags: [python, word cloud]
---

### 抓取关键字搜索 

现在有很多的工具和已成型的代码，用于在网络上爬虫抓取内容，进行情感分析或者趋势分析。这里的代码是参考的这里[gaussic/weibo_wordcloud @github](https://github.com/gaussic/weibo_wordcloud)，尝试着抓取数据并对内容进行分词分析。

抓取：

```python
# coding: utf-8

import re
import json
import requests

# 基于 m.weibo.cn 抓取少量数据，无需登陆验证
url_template = "https://m.weibo.cn/api/container/getIndex?type=wb&queryVal={}&containerid=100103type=2%26q%3D{}&page={}"


def clean_text(text):
    """清除文本中的标签等信息"""
    dr = re.compile(r'(<)[^>]+>', re.S)
    dd = dr.sub('', text)
    dr = re.compile(r'#[^#]+#', re.S)
    dd = dr.sub('', dd)
    dr = re.compile(r'@[^ ]+ ', re.S)
    dd = dr.sub('', dd)
    return dd.strip()


def fetch_data(query_val, page_id):
    """抓取关键词某一页的数据"""
    resp = requests.get(url_template.format(query_val, query_val, page_id))
    card_group = json.loads(resp.text)['data']['cards'][0]['card_group']
    print('url：', resp.url, ' --- 条数:', len(card_group))

    mblogs = []  # 保存处理过的微博
    for card in card_group:
        mblog = card['mblog']
        blog = {'mid': mblog['id'],  # 微博id
                'text': clean_text(mblog['text']),  # 文本
                'userid': str(mblog['user']['id']),  # 用户id
                'username': mblog['user']['screen_name'],  # 用户名
                'reposts_count': mblog['reposts_count'],  # 转发
                'comments_count': mblog['comments_count'],  # 评论
                'attitudes_count': mblog['attitudes_count']  # 点赞
                }
        mblogs.append(blog)
    return mblogs


def remove_duplication(mblogs):
    """根据微博的id对微博进行去重"""
    mid_set = {mblogs[0]['mid']}
    new_blogs = []
    for blog in mblogs[1:]:
        if blog['mid'] not in mid_set:
            new_blogs.append(blog)
            mid_set.add(blog['mid'])
    return new_blogs


def fetch_pages(query_val, page_num):
    """抓取关键词多页的数据"""
    mblogs = []
    for page_id in range(1 + page_num + 1):
        try:
            mblogs.extend(fetch_data(query_val, page_id))
        except Exception as e:
            print(e)

    print("去重前：", len(mblogs))
    mblogs = remove_duplication(mblogs)
    print("去重后：", len(mblogs))

    # 保存到 result.json 文件中
    fp = open('result_{}.json'.format(query_val), 'w', encoding='utf-8')
    json.dump(mblogs, fp, ensure_ascii=False, indent=4)
    print("已保存至 result_{}.json".format(query_val))


if __name__ == '__main__':
    fetch_pages('颜宁', 50)
```

抓取搜索`颜宁`的结果：

```bash
% /Users/gongjing/anaconda3/bin/python weibo_search.py

去重前： 454
去重后： 443
已保存至 result_颜宁.json
```

使用jieba分词，并画出词云图：

```python
# coding: utf-8

import json

import jieba.analyse
import matplotlib as mpl
# from scipy.misc import imread
from imageio import imread
from wordcloud import WordCloud

# mpl.use('TkAgg')
import matplotlib.pyplot as plt


def keywords(mblogs):
    text = []
    for blog in mblogs:
        keyword = jieba.analyse.extract_tags(blog['text'])
        text.extend(keyword)
    return text


def gen_img(texts, img_file):
    data = ' '.join(text for text in texts)
    image_coloring = imread(img_file)
    wc = WordCloud(
        background_color='white',
        mask=image_coloring,
        font_path='/Library/Fonts/STHeiti Light.ttc'
    )
    wc.generate(data)

    # plt.figure()
    # plt.imshow(wc, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()

    wc.to_file(img_file.split('.')[0] + '_wc.png')


if __name__ == '__main__':
    keyword = '颜宁'
    mblogs = json.loads(open('result_{}.json'.format(keyword), 'r', encoding='utf-8').read())
    print('微博总数：', len(mblogs))

    words = []
    for blog in mblogs:
        words.extend(jieba.analyse.extract_tags(blog['text']))

    print("总词数：", len(words))

    gen_img(words, 'yanning.jpg')
```

```bash
% /Users/gongjing/anaconda3/bin/python weibo_cloud.py

微博总数： 443
Building prefix dict from the default dictionary ...
Dumping model to file cache /var/folders/1c/bxl334sx7797nfp66vmrphxc0000gn/T/jieba.cache
Loading model cost 1.174 seconds.
Prefix dict has been built succesfully.
总词数： 6862
/Users/gongjing/anaconda3/bin/python weibo_cloud.py  4.07s user 0.98s system 80% cpu 6.305 total
```

生成的词云和颜宁老师的日常很像（尤其是朱一龙这个），但是因为给出的背景图辨识度不高，所以从词云里面不能一眼看不出就是她：

[![weibo_yanning.jpeg](https://i.loli.net/2019/07/12/5d28a5840279576944.jpeg)](https://i.loli.net/2019/07/12/5d28a5840279576944.jpeg)
