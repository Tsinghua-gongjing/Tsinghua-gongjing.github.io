---
layout: post
category: "visualization"
title:  "Violin plot"
tags: [plot, visualization]
---

### plot use `sns.violinplot`

```python
# violin plot
def df_sns_violinplot(df,col_str,savefn,orient='v',class_col="",order=None):
    assert savefn.endswith("png"), "[savefn error] should be .png: %s"%(savefn)
    savefn=savefn.replace('png','violin.png')

    print "[df_sns_violinplot]"
    col_str = None if col_str == "" else col_str
    class_col= None if class_col == "" else class_col
    print "  - col_str: %s"%(col_str)
    print "  - class_col: %s"%(class_col)
    print df.head(2)
    print df.dtypes

    #fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(8,8)) # plot multiple violin in one fig
    if class_col:
        s=sns.violinplot(y=col_str,x=class_col,data=df,order=order) # ax=ax[0,1,...]
    else:
        print "plot col_str: ",col_str
        #s=sns.violinplot(x='p_val_poisson',data=df) # ax=ax[0,1,...]
        s=sns.violinplot(df[col_str],order=order) # ax=ax[0,1,...]

    title_str=savefn.split("/")[-1]+':'+str(df.shape[0])
    fig=s.get_figure()
    fig.suptitle(title_str)
    fig.savefig(savefn)
    print "  - savefig: %s"%(savefn)
    plt.close()
    return savefn
```

