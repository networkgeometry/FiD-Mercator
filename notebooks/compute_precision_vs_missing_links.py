import numpy as np

def compute_precision(ET, EP, L, Score):
    """
    Function to calculate the precision of the link prediction method

    ET: NxN adjacency matrix, representing training data
    EP: NxN adjanceny matrix, representing testing data
    L: number of removed links (or the number of links to predict)
    Score: NxN matrix where entries represent the similarity scores between each pair of nodes 
    """
    Score1 = Score.copy()
    # Find indices of the existing links
    Ind_edg = np.argwhere(ET > 0)
    
    # We set the entries of the score matrix corresponding to the existing links to a very small value
    # (so we push the existing links to the end of the list of the links ordered in decreasing order of their scores)
    Score1[tuple(np.transpose(Ind_edg))] = -1000 # Lower that the mimum value in the score matrix
    np.fill_diagonal(Score1, -1000) # To ignore self loops
    
    # Set the entries of the score matrix in the lower triangle 
    # to a very small value (in order not to put a nonexisting link with a high score at the beginning part of the ordered list twice)
    Score1[np.tril_indices(Score1.shape[0])] = -1000
    
    # Set the entires in the lower triangle part of EP to zero (to count each link only one time)
    EP[np.tril_indices(EP.shape[0])] = 0
 
    # To sort the edges according to their scores
    ASc_list = np.dstack(np.unravel_index(np.argsort(Score1, axis=None)[::-1], Score1.shape))[0]

    # To select L links with the highest score as the predicted links
    # New_Edges keeps the index of nonexisting links with the highest score
    New_Edges = ASc_list[:L]

    # To check if the predicted links are in the probe set
    if len(New_Edges) > 0:
        Ind_prb = np.argwhere(EP[tuple(np.transpose(New_Edges))] > 0)
        Lr = len(Ind_prb)
        return Lr / L
    else:
        return 0