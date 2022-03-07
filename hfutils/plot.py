from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

def sns_displot(d, label, **kwds):
    sns.distplot(d, hist=True, kde=False, 
        bins=int(180/5), 
        # color = 'darkblue', 
        label=label,
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 2}, **kwds)

def distplot(path, data, labels=None, **kwds):
    if isinstance(data, (List, np.ndarray)):
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.reshape((data.shape[0], -1))
            for i, d in enumerate(data):
                sns_displot(d, None if labels is None else labels[i], **kwds)
        else:
            sns_displot(data, None if labels is None else labels, **kwds)

    if isinstance(data, Dict):
        for k, v in data.items():
            sns_displot(v, k, **kwds)

    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()