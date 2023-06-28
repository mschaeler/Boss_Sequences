import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



RESULT_FILE_LOC = "./baseline/results/esv_king_james_{0}_k{1}_0.7.txt"

def get_windows(s, k):
    s = s.split(" ")
    s = [s1.strip() for s1 in s]
    return [s[i:i+k] for i in range(len(s) - (k - 1))]



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def plot_heatmap():
    S = "in those days when king Ahasuerus sat on his royal throne in Susa, the citadel"
    T = "That in those days, when the king Ahasuerus sat on the throne of his kingdom, which was in Shushan the palace"
    k = 3
    theta = 0.3
    W_S = get_windows(S, k)
    W_S = [" ".join(s) for s in W_S]
    W_T = get_windows(T, k)
    W_T = [" ".join(t) for t in W_T]

    A = np.zeros((len(W_S), len(W_T)))

    for i in range(len(W_S)):
        for j in range(len(W_T)):
            sim = jaccard_similarity(W_S[i], W_T[j])
            if (sim >= theta):
                A[i][j] = sim
            
    
    print(A)

    labels_ws = ["S_{0}".format(i) for i in range(1, len(W_S) + 1)]
    labels_wt = ["T_{0}".format(i) for i in range(1, len(W_T) + 1)]

    fig, ax = plt.subplots()
    im, cbar = heatmap(A, labels_ws, labels_wt, ax=ax, cmap="YlGn", cbarlabel="alignment matrix [W_S/W_T]")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    fig.tight_layout()
    # plt.show()
    plt.savefig("jaccard_heatmap.png")
    plt.cla()
    plt.clf()




def get_data(fileloc):
    file = open(fileloc, 'r')
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    file.close()
    result_data = {}
    for line in lines:
        if line.startswith('Inverted Index Size:'):
            s = line.split(':')
            result_data['invertedIndex'] = int(s[1].strip())
        
        if line.startswith('Size of valid edges:'):
            s = line.split(':')
            result_data['validEdges'] = int(s[1].strip())
        
        if line.startswith('Number of Graph Matching Computed:'):
            s = line.split(':')
            result_data['alignmentMatrixSize'] = int(s[1].strip())

        if line.startswith('Number of Zero Entries Cells:'):
            s = line.split(':')
            result_data['belowThresholdCells'] = int(s[1].strip())

        if line.startswith('Env Time:'):
            s = line.split(':')
            result_data['envTime'] = float(s[1].strip())

        if line.startswith('Dataloader Time:'):
            s = line.split(':')
            result_data['dataloaderTime'] = float(s[1].strip())

        if line.startswith('FaissIndex Time:'):
            s = line.split(':')
            result_data['faissTime'] = float(s[1].strip())

        if line.startswith('Algorithm Time:'):
            s = line.split(':')
            result_data['algoTime'] = float(s[1].strip())

    return result_data


def plot_results():
    for k in [3, 5, 7]:
        algo_time = []
        alignmentSize = []
        zeroCells = []
        for para in range(1, 11):
            file_name = RESULT_FILE_LOC.format(para, k)
            data = get_data(file_name)
            algo_time.append(data['algoTime'])
            alignmentSize.append(data['alignmentMatrixSize'])
            zeroCells.append(data['belowThresholdCells'])
        
        x = [i for i in range(1, 11)]
        plt.plot(x, algo_time, '-o')
        plt.xticks(x)
        plt.xlabel('Paragraphs')
        plt.ylabel('Algorithm Time (seconds)')
        plt.tight_layout()
        plt.savefig('./baseline/plots/algoTime_k{0}.png'.format(k))
        plt.cla()
        plt.clf()

        above_threshold = [x - y for x,y in zip(alignmentSize, zeroCells)]
        ind = np.arange(len(x))
        width = 0.2
        plt.bar(ind, zeroCells, width, label='below threshold')
        plt.bar(ind + width, above_threshold, width, label='above threshold')
        plt.xlabel('Paragraphs')
        plt.ylabel('Number of Cells (log scale)')
        plt.yscale('log')
        plt.xticks(ind + width /2, x)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./baseline/plots/alignmentMatrixBreakdown_k{0}.png'.format(k))
        plt.cla()
        plt.clf()



if __name__ == "__main__":
    # plot_heatmap()
    plot_results()