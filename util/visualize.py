from turtle import color
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from matplotlib import pyplot as plt
from matplotlib import cm

COLOR = {
    'location': ['red', 'lightcoral', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred',],
    'organization': ['green', 'lightgreen', 'forestgreen', 'limegreen', 'darkgreen', 'lime', 'seagreen', 'lawngreen', 'greenyellow', 'yellowgreen'],
    # 'other': ['violet', 'darkmagenta', 'm', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink','palevioletred', 'crimson', 'pink', 'lightpink'],
    # 'person': ['orange', 'coral', 'tomato', 'salmon', 'lightsalmon', 'sandybrown', 'darkorange', 'wheat'],
    # 'product': ['yellow', 'gold', 'khaki', 'olive', 'y', 'goldenrod', 'darkgoldenrod', 'darkkhaki', 'lemonchiffon'],
    # 'art': ['cyan', 'aqua', 'c', 'darkcyan', 'teal', 'paleturquoise'],
    # 'building': ['purple', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet'],
    # 'event': ['blue', 'mediumblue', 'darkblue', 'navy', 'midnightblue', 'royalblue', 'cornflowerblue'],
    'O': ['lightgrey']
}

tot = np.sum([len(v) for v in COLOR.values()])
col_list = np.linspace(0, 1, tot)
col = {}
tmp = 0
for k, v in COLOR.items():
    col[k] = cm.rainbow(col_list[tmp: tmp + len(v)])
    tmp += len(v)

def visualize(X, y, prefix='vis/', save=True):
    if save:
        torch.save({'X':X, 'y':y}, f'{prefix}data.pt')
    v_methods = [
        PCA(n_components=2),
        TruncatedSVD(n_components=2, n_iter=7, random_state=42),
        TSNE(n_components=2, init='random')]
    d = set(y)
    d = list(d)
    d.sort()
    print(d)

    for i, method in enumerate(v_methods):
        X_embedded = method.fit_transform(X)
        cnt = {i:0 for i in COLOR.keys()}
        for label in d:
            if label == 'O':
                coarse = 'O'
                fine = ''
            else:
                coarse, fine = label.split('-')
            mask = np.array([a==label for a in y])
            
            print(coarse, fine, cnt[coarse])
            # plt.scatter(X_embedded[mask,0],X_embedded[mask,1], label=label, s=1, alpha=1,c=COLOR[coarse][cnt[coarse]],edgecolors=COLOR[coarse][0],linewidths=.1)
            if label != 'O':
                plt.scatter(X_embedded[mask,0], X_embedded[mask,1], label=label, s=1, color=col[coarse][cnt[coarse]])
            else:
                plt.scatter(X_embedded[mask,0], X_embedded[mask,1], label=label, s=1, c='lightgrey')
            cnt[coarse] += 1
        # plt.legend()
        # plt.title(str(method))
        plt.savefig(f'{prefix}{i}.png',dpi=500)
        plt.close()

if __name__=="__main__":
    # for part in ['train', 'val', 'test']:
    #     data = torch.load(f'vis/CONTAINERvis_test_withoutO_data.pt')
    #     X = data['X']
    #     y = data['y']
    #     visualize(X, y, prefix=f'vis/{part}_',save=False)

    data = torch.load(f'vis/CONTAINERvis_test_withoutO_data.pt')
    X = data['X']
    y = data['y']
    visualize(X, y, prefix=f'vis/CONTAINERvis_test_withoutO_alter_',save=False)
