---
layout: post
category: "other"
title:  "Build personal public blog based on Github Pages"
tags: [github, blog]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

- TOC
{:toc}

## 搭建流程

这个博客的搭建主要是基于github pages，使用jekyll模板。在线的流程有很多，我主要是参考了[这篇](http://www.abstractclass.org/tutorial/blog/2015/05/19/tutorial-personal-blog-with-github.html)文章。

简要的步骤如下：

* 开通github账号
* 新建仓库（username.github.io）；最好是直接fork一个自己喜欢的模板，然后把仓库名称改为这个。我这里使用的是从[这里](https://github.com/panxw/panxw.github.com)fork的，非常感谢。[jekyll主题网](https://github.com/jekyll/jekyll/wiki/Themes)也提供了很多不同的主题，可以选择自己喜欢的，下载下来。
* 在本地clone下来
* 本地安装jekyll（静态网站生成的工具，基于ruby）
* 配置_config.yml文件
* 本地查看，在本地仓库目录下执行：jekyll serve
* 如果觉得效果还可以，推送到远程

## 申请学生账户

参考博客[（GitHub教程 学生认证——学生包申请）](https://blog.csdn.net/qq_36667170/article/details/79084166)进行申请，因为用的是教育邮箱，所以不到两分钟就收到了通过邮件。

```
Hey XXX, we have some awesome news

We've upgraded you to a plan with unlimited free private repositories, which will be free for the next two years. After that, you'll get an email saying that your coupon is expiring. You can reapply for another coupon if you still have academic status. We don't have any collaboration limits, so any group projects you may encounter can be hosted via your account.

If you need help getting started with Git and GitHub, check out:

https://help.github.com/articles/good-resources-for-learning-git-and-github

We've also given you access to the Student Developer Pack, available at:

https://education.github.com/pack

If you have any questions, contact us:

https://education.github.com/contact

Spread the word: we love giving educational discounts to students, teachers, administrators, and researchers! Please send them to:

https://education.github.com

Have an Octotastic day!
- The GitHub Education Team 
```

## 成形文件

本地的文件目录大致如下：

~~~ bash
gongjing@hekekedeiMac ~/Dropbox/Tsinghua-gongjing.github.io (git)-[master] % ll
total 52K
-rw-r--r--  1 gongjing staff  156 Feb  5 21:55 404.html
-rw-r--r--  1 gongjing staff    0 Feb  5 21:55 CNAME
-rw-r--r--  1 gongjing staff  654 Feb  7 13:53 README.md
-rw-r--r--  1 gongjing staff  852 Feb  7 22:25 _config.yml
drwxr-xr-x 12 gongjing staff  408 Feb  7 20:37 _includes
drwxr-xr-x  7 gongjing staff  238 Feb  8 10:48 _layouts
drwxr-xr-x 37 gongjing staff 1.3K Feb  9 10:37 _posts
drwxr-xr-x 19 gongjing staff  646 Feb  8 15:41 _site
-rw-r--r--  1 gongjing staff 2.5K Feb  9 10:19 about.md
-rw-r--r--  1 gongjing staff  579 Feb  5 21:55 archives.md
drwxr-xr-x  8 gongjing staff  272 Feb  6 20:04 assets
-rw-r--r--  1 gongjing staff  743 Feb  5 21:55 atom.xml
-rw-r--r--  1 gongjing staff  422 Feb  5 21:55 categories.md
drwxr-xr-x  4 gongjing staff  136 Feb  7 18:43 css
-rw-r--r--  1 gongjing staff 1.1K Feb  5 21:55 faqs.md
-rw-r--r--  1 gongjing staff 1.2K Feb  5 21:55 favicon.ico
drwxr-xr-x  3 gongjing staff  102 Feb  5 21:55 fonts
-rw-r--r--  1 gongjing staff  945 Feb  6 11:11 index.html
drwxr-xr-x  6 gongjing staff  204 Feb  5 21:55 js
-rw-r--r--  1 gongjing staff  679 Feb  5 21:55 links.md
drwxr-xr-x  8 gongjing staff  272 Feb  6 11:44 posts
-rw-r--r--  1 gongjing staff   53 Feb  5 21:55 robots.txt
-rw-r--r--  1 gongjing staff 3.9K Feb  5 21:55 tags.md
~~~

这路使用的markdown文件来写post（放在\_posts目录下面），github pages会基于jekyll自动生成网页的格式。_layout文件夹定义了每个网页的基本格式，可以通过修改这里的html文件，调整网页的布局。

## 添加评论

clone的这个模板没有评论部分的代码，看了一下原来的网站，使用的gitment服务（评论需要github账号）。另一个用的比较多的是Disqus服务，于是自己添加代码搭建了一个。主要的参看是[这里](https://poanchen.github.io/blog/2017/07/27/how-to-add-disqus-to-your-jekyll-site)。

注意：

* 在_config.yml配置文件中，disqus shortname是网站的（**不是自己的用户的**）: 比如我的是https-tsinghua-gongjing-github-io
* disqus功能需要翻墙

## 图片托管

博客会有很多图片，使得传达信息更加直接，为了使自己的网站不臃肿，可以把图片都放在一些图床上，然后在post中放图片的链接，生成网页时会直接加载。目前正在申请[七牛云](https://www.qiniu.com/)的账号，免费账号10G存储（需要拿着身份证拍正反面照片，上传）。

**更新**：对于图片直接放在assets(比如：/Users/gongjing/Dropbox/Tsinghua-gongjing.github.io/assets)目录下面即可，然后在md文件写相对链接(因为jekyll会自动解析相对网站域名的路径， 可参考[这里](https://sgeos.github.io/github/jekyll/2016/08/30/adding_images_and_downloads_to_a_github_pages_jekyll_blog.html)，PDF文件也可以直接放在这个目录下，在浏览器中直接打开或者下载)，比如：

```
![bed](/assets/bed_file_format_example.jpeg)
```

## 更换网站tab图片

设计一个图片后，上传到[favicon](http://www.favicon.cc/)生成，对应的icon文件，放在root目录下即可。（如果更换后，加载没有更新，需要清理一下网站的缓存）


## 在kramdown中添加LATEX

目前实现的是在单个post(.md文件)中添加，需要在开头显式指定调用[MathJax](http://www.mathjax.org/)，因为其本身不提供解析。kramdown的[math-block](https://kramdown.gettalong.org/syntax.html#math-blocks)，参考[github render math blocks](https://github.com/mmistakes/minimal-mistakes/issues/735)。
比如在写.md文件时，先调用，效果参见[post](https://tsinghua-gongjing.github.io//posts/ThinkStats_descriptive_statistic.html)。

```
---
layout: post
category: "read"
title:  "Think Stats: descriptive statistics?"
tags: [reading, statistics]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

1. 均值（mean）：值的总和除以值的数量；平均值（average）：若干种可以用于描述样本的典型值或**集中趋势（central tendency）**的汇总统计量之一。注意根据样本的范围选择合适的描述量。
2. 方差：描述分散情况。

$$
 \begin{align}\sigma^2 = \frac{1}{n}∑(X_i-\mu)^2\end{align} 
$$
```

## 在kramdown中添加html表格，带图片的

```html
<div>
<table>
  <tr>
    <th align="middle">Not smooth</th>
    <th align="middle">Smooth</th>
  </tr>
  <tr>
    <td><img src="https://i.stack.imgur.com/dSLtt.png"></td>
    <td><img src="https://i.stack.imgur.com/olGAh.png"></td>
  </tr>
</table>
</div>
```

## 网址显示乱码的问题

在每个页面下方，有个链接，但是当链接含有中文时，显示出一长串的码，所以需要解析：

post页面地址：

```bash
https://tsinghua-gongjing.github.io/posts/模型评估与选择.html

```

显示：

```bash
If you link this blog, please refer to this page, thanks! 
Post link：https://tsinghua-gongjing.github.io/posts/%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E4%B8%8E%E9%80%89%E6%8B%A9.html

```

在`/_layouts/post.html`页面的代码部分：

```bash
<div style="margin:10px">
<br/>
<b>If you link this blog, please refer to this page, thanks!</b>
<br/>
<b>Post link：</b><a href="\\{{site.url}}{{page.url}}" title="{{page.name}}">{{site.url}}{{page.url}}</a>
</div>
```
