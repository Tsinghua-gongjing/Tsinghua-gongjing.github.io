---
layout: post
category: "python"
title:  "Python version"
tags: [plot]
---

## install tensorflow-gpu

之前安装了Anaconda3，是3.7版本的。最近需要使用到`tensorflow-gpu`，就用`pip`安装了一下，是最新的1.13.1版本的。

```bash
pip install tensorflow-gpu
```

使用的时候会报错：

```python
ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory
```

这是因为系统安装的`Cuda`版本不符合`tensorflow-gpu`的要求，所以按照[这里](https://github.com/tensorflow/tensorflow/issues/26209)，想要安装低版本的`tensorflow-gpu`。

```bash
$ pip install tensorflow-gpu==1.12.0
Collecting tensorflow-gpu==1.12.0
  Could not find a version that satisfies the requirement tensorflow-gpu==1.12.0 (from versions: 1.13.0rc1, 1.13.0rc2, 1.13.1, 2.0.0a0)
No matching distribution found for tensorflow-gpu==1.12.0
```

不能通过`pip`安装，想着能不能直接下载文件安装，所以参考[这里](https://stackoverflow.com/questions/27885397/how-do-i-install-a-python-package-with-a-whl-file)直接下载模块对应的`whl`文件，然后使用`pip`命令安装：

```bash
$ wget https://files.pythonhosted.org/packages/e3/d3/4a356db5b6a2c9dcb30011280bc065cf51de1e4ab5a5fee44eb460a98449/tensorflow_gpu-1.4.1-cp36-cp36m-manylinux1_x86_64.whl

$ pip install tensorflow_gpu-1.4.1-cp36-cp36m-manylinux1_x86_64.whl
tensorflow_gpu-1.4.1-cp36-cp36m-manylinux1_x86_64.whl is not a supported wheel on this platform.
```

可以看到安装说平台不兼容，是因为这是对应于`python3.6`版本的，而我的python是3.7版本的，[这里](https://pypi.org/project/tensorflow-gpu/1.4.1/#files)列举的`tensorflow_gpu-1.4.1`支持的python版本有`2.7`,`3.3`,`3.4`,`3.5`,`3.6`，没有`3.7`的，所以即使下载了`whl`文件，也不能正常安装。

所以接下来就降低自己的python版本，从`3.7`降到`3.6`,参考[这里](https://stackoverflow.com/questions/52584907/how-to-downgrade-python-from-3-7-to-3-6)使用`conda`直接进行，不仅更新了python，也更新了对应的其他模块。

```bash
$ conda install python=3.6.0
Collecting package metadata: done
Solving environment: done

## Package Plan ##

  environment location: /home/gongjing/software/anaconda3

  added / updated specs:
    - python=3.6.0


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    _ipyw_jlab_nb_ext_conf-0.1.0|           py36_0           4 KB
    alabaster-0.7.12           |           py36_0          17 KB
    anaconda-client-1.7.2      |           py36_0         141 KB
    anaconda-navigator-1.9.7   |           py36_0         4.7 MB
    anaconda-project-0.8.2     |           py36_0         478 KB
    asn1crypto-0.24.0          |           py36_0         155 KB
    
pep8-1.7.1           | 52 KB     | ################################################################################################################### | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

降低了python的版本后，就可以直接使用pip安装低版本的`tensorflow-gpu`了。

```bash
(base) [gongjing@localhost ~]$ pip install tensorflow_gpu==1.4.1
Collecting tensorflow_gpu==1.4.1
  Downloading https://files.pythonhosted.org/packages/e3/d3/4a356db5b6a2c9dcb30011280bc065cf51de1e4ab5a5fee44eb460a98449/tensorflow_gpu-1.4.1-cp36-cp36m-manylinux1_x86_64.whl (170.3MB)
     |████████████████████████████████| 170.3MB 956kB/s
Requirement already satisfied: numpy>=1.12.1 in /data01/gongjing/software/anaconda3/lib/python3.6/site-packages (from tensorflow_gpu==1.4.1) (1.15.0)
Requirement already satisfied: six>=1.10.0 in /data01/gongjing/software/anaconda3/lib/python3.6/site-packages (from tensorflow_gpu==1.4.1) (1.12.0)
Collecting protobuf>=3.3.0 (from tensorflow_gpu==1.4.1)
  Downloading https://files.pythonhosted.org/packages/5a/aa/a858df367b464f5e9452e1c538aa47754d467023850c00b000287750fa77/protobuf-3.7.1-cp36-cp36m-manylinux1_x86_64.whl (1.2MB)
     |████████████████████████████████| 1.2MB 2.0MB/s
Requirement already satisfied: wheel>=0.26 in /data01/gongjing/software/anaconda3/lib/python3.6/site-packages (from tensorflow_gpu==1.4.1) (0.33.1)
Collecting enum34>=1.1.6 (from tensorflow_gpu==1.4.1)
  Downloading https://files.pythonhosted.org/packages/af/42/cb9355df32c69b553e72a2e28daee25d1611d2c0d9c272aa1d34204205b2/enum34-1.1.6-py3-none-any.whl
Collecting tensorflow-tensorboard<0.5.0,>=0.4.0rc1 (from tensorflow_gpu==1.4.1)
  Downloading https://files.pythonhosted.org/packages/e9/9f/5845c18f9df5e7ea638ecf3a272238f0e7671e454faa396b5188c6e6fc0a/tensorflow_tensorboard-0.4.0-py3-none-any.whl (1.7MB)
     |████████████████████████████████| 1.7MB 1.1MB/s
Requirement already satisfied: setuptools in /data01/gongjing/software/anaconda3/lib/python3.6/site-packages (from protobuf>=3.3.0->tensorflow_gpu==1.4.1) (41.0.1)
Collecting bleach==1.5.0 (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow_gpu==1.4.1)
  Downloading https://files.pythonhosted.org/packages/33/70/86c5fec937ea4964184d4d6c4f0b9551564f821e1c3575907639036d9b90/bleach-1.5.0-py2.py3-none-any.whl
Collecting html5lib==0.9999999 (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow_gpu==1.4.1)
  Downloading https://files.pythonhosted.org/packages/ae/ae/bcb60402c60932b32dfaf19bb53870b29eda2cd17551ba5639219fb5ebf9/html5lib-0.9999999.tar.gz (889kB)
     |████████████████████████████████| 890kB 1.8MB/s
Collecting markdown>=2.6.8 (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow_gpu==1.4.1)
  Using cached https://files.pythonhosted.org/packages/f5/e4/d8c18f2555add57ff21bf25af36d827145896a07607486cc79a2aea641af/Markdown-3.1-py2.py3-none-any.whl
Requirement already satisfied: werkzeug>=0.11.10 in /data01/gongjing/software/anaconda3/lib/python3.6/site-packages (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow_gpu==1.4.1) (0.15.2)
Building wheels for collected packages: html5lib
  Building wheel for html5lib (setup.py) ... done
  Stored in directory: /home/gongjing/.cache/pip/wheels/50/ae/f9/d2b189788efcf61d1ee0e36045476735c838898eef1cad6e29
Successfully built html5lib
Installing collected packages: protobuf, enum34, html5lib, bleach, markdown, tensorflow-tensorboard, tensorflow-gpu
  Found existing installation: html5lib 1.0.1
    Uninstalling html5lib-1.0.1:
      Successfully uninstalled html5lib-1.0.1
  Found existing installation: bleach 3.1.0
    Uninstalling bleach-3.1.0:
      Successfully uninstalled bleach-3.1.0
Successfully installed bleach-1.5.0 enum34-1.1.6 html5lib-0.9999999 markdown-3.1 protobuf-3.7.1 tensorflow-gpu-1.4.1 tensorflow-tensorboard-0.4.0
```