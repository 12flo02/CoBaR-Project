from sklearn.manifold import TSNE


def runTSNE(wavelet_array, groupByFly=False):
    perplexity = 30 if groupByFly else 50
    
    return TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(wavelet_array)