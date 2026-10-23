import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform


def data(path, non_numerical_var, expath):
    import pandas as pd
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    #Cargo y elimino variables categoricas
    data = pd.read_csv(path)
    numeric_data = data.drop(non_numerical_var, axis =1)

    # Normalizo datos
    norm_data = (numeric_data - numeric_data.mean())/numeric_data.std()

    #Obtengo matriz de distancias
    distances = pdist(norm_data.head(20).values, metric='euclidean')
    dist_matrix = squareform(distances)

    #Exporto Matriz como archivo .npy
    np.fill_diagonal(dist_matrix, 1e9)
    np.save(expath, dist_matrix)
    #np.save('Distances', dist_matrix)