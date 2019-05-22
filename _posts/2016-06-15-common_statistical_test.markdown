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

#### test distribution

```python
# test based on poisson distribution
def sig_test_poisson(x=0.3,mu=0.001):
    p=stats.distributions.poisson.pmf(x,mu)
    #print "poisson test:",p # work correctly
    return p

# test based on negative binomial distribution
def sig_test_nbinom(x=100,n=50,p=0.3):
    p=stats.distributions.nbinom.pmf(x,n,p)
    #print "nbinom test:",p # work correctly
    return p

```

#### KS test

```python
# two sample distribution test
def ks_2samp(x,y):
    p=stats.ks_2samp(x,y)[1]
    return p
```

#### calculate correlation

```python
# two sample rank test
def sig_spearman_corr(x,y):
    p=stats.spearmanr(x,y)[0]
    return p

def sig_pearson_corr(x,y):
    p=stats.pearsonr(x,y)[0]
    return p
```