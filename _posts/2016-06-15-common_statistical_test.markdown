---
layout: post
category: "statistics"
title:  "Common test to measure significance"
tags: [statistics]
---

### Different types of hypothesis test

From Python for R users, p151

![img](/assets/types_of_hypothesis_test.jpeg)

### Test in Python Scipy stat

#### [fisher exact test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)

```python
import scipy.stats as stats
oddsratio, pvalue = stats.fisher_exact([[8, 2], [1, 5]])
print oddsratio,pvalue

# 20.0 0.03496503496503495
```