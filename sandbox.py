# Import downloaded modules
import numpy as np
import tensorflow as tf
import spektral

# Download CORA dataset and its different members
dataset = spektral.datasets.citation.Citation(
    'cora', 
    random_split=False, # split randomly: 20 nodes per class for training, 30 nodes 
        # per class for validation; or "Planetoid" (Yang et al. 2016)
    normalize_x=False,  # normalize the features
    dtype=np.float32 # numpy data type for the graph data
    )
dataset.graphs[0]

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

pass

