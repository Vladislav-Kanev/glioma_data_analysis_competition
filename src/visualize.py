import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


def make_2d_representation(dataset: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(8, 4))
    axe = fig.gca()

    np_dataset = dataset.to_numpy()

    tsne = TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(np_dataset)
    axe.scatter(transformed_data[:, 0], transformed_data[:, 1])
    axe.set_title('TSNE data representation')
    fig.set_dpi(300)
    return fig
