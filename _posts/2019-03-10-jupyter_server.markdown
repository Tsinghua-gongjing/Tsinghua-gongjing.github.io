---
layout: post
category: "python"
title:  "Connect remote jupyter server"
tags: [jupyter]
---

## 连接使用远程（集群）jupyter notebook

主要是参考了[这里](http://danielhnyk.cz/running-ipython-notebook-different-computer/)，这个是用的`ipython`，当时`jupyter notebook`还没完善，基本原理和操作是一样的，包括现在很多人使用的`jupyter-lab`，下面是连接`jupyter-lab`的例子：

1、在客户端打开一个指定端口的server：
   
```bash
jupyter-lab --port 8007
```

```
[W 13:37:55.064 LabApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
[I 13:37:55.378 LabApp] JupyterLab beta preview extension loaded from /Share/home/zhangqf5/anaconda2/lib/python2.7/site-packages/jupyterlab
[I 13:37:55.378 LabApp] JupyterLab application directory is /Share/home/zhangqf5/anaconda2/share/jupyter/lab
[I 13:37:55.389 LabApp] Serving notebooks from local directory: /Share/home/zhangqf5/gongjing/rare_cell
[I 13:37:55.389 LabApp] 0 active kernels
[I 13:37:55.389 LabApp] The Jupyter Notebook is running at:
[I 13:37:55.389 LabApp] http://[all ip addresses on your system]:8007/?token=442494e19571582585464518668825f6ae1e4d4c3bdfc070
[I 13:37:55.389 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 13:37:55.391 LabApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8007/?token=442494e19571582585464518668825f6ae1e4d4c3bdfc070
[I 13:42:16.820 LabApp] 302 GET / (::1) 0.47ms
[W 13:42:18.565 LabApp] Could not determine jupyterlab build status without nodejs
```

2、在本地使用`ssh`用指定端口登录：

```bash
ssh -Y username@remoteIP -L 8007:localhost:8007
```

3、在本地浏览器输入对应的地址：`http://localhost:8007`，有的时候需要指定一个`token`，所以可以直接复制完整第地址，比如这里的:

  `http://localhost:8007/?token=442494e19571582585464518668825f6ae1e4d4c3bdfc070`

