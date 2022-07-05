# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 21:42:20 2021

@author: Daniel
"""

#####This code has been taken with courtesy from:
#https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/tree/master/Recommenders
import numpy as np
import time, sys, warnings
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
import shutil
def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """


    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)

    elif format == 'npy':
        if sps.issparse(X):
            return X.toarray().astype(dtype)
        else:
            return np.array(X)

    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)


def similarityMatrixTopK(item_weights, k=100, verbose = False):
    """
    The function selects the TopK most similar elements, column-wise
    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    # iterate over each column and keep only the top-k similar items
    data, rows_indices, cols_indptr = [], [], []

    if sparse_weights:
        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)
    else:
        column_row_index = np.arange(nitems, dtype=np.int32)



    for item_idx in range(nitems):

        cols_indptr.append(len(data))

        if sparse_weights:
            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx+1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

        else:
            column_data = item_weights[:,item_idx]


        non_zero_data = column_data!=0

        idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
        top_k_idx = idx_sorted[-k:]

        data.extend(column_data[non_zero_data][top_k_idx])
        rows_indices.extend(column_row_index[non_zero_data][top_k_idx])


    cols_indptr.append(len(data))

    # During testing CSR is faster
    W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)

    if verbose:
        print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

    return W_sparse

def seconds_to_biggest_unit(time_in_seconds, data_array = None):

    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value/conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            if data_array is not None:
                data_array /= conversion_factor[unit_index][1]

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True


    if data_array is not None:
        return new_time_value, new_time_unit, data_array

    else:
        return new_time_value, new_time_unit

class Compute_Similarity_Python:


    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize = True,
                 asymmetric_alpha = 0.5, tversky_alpha = 1.0, tversky_beta = 1.0,
                 similarity = "cosine", row_weights = None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions
        """
        """
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
        
        """

        super(Compute_Similarity_Python, self).__init__()


        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topK, self.n_columns)

        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for parameter 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))


        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Col_weights has {} columns, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T






    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csr')


        interactionsPerRow = np.diff(self.dataMatrix.indptr)

        nonzeroRows = interactionsPerRow > 0
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sumPerRow)
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]


        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row= 0

        blockSize = 1000


        while end_row < self.n_rows:

            end_row = min(self.n_rows, end_row + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize




    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')


        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]


        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col= 0

        blockSize = 1000


        while end_col < self.n_columns:

            end_col = min(self.n_columns, end_col + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize


    def useOnlyBooleanInteractions(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos= 0

        blockSize = 1000


        while end_pos < len(self.dataMatrix.data):

            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos-start_pos)

            start_pos += blockSize




    def compute_similarity(self, start_col=None, end_col=None, block_size = 100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0


        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.useOnlyBooleanInteractions()


        # We explore the matrix column-wise
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')


        # Compute sum of squared values to be used in normalization
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sumOfSquared = np.sqrt(sumOfSquared)

        if self.asymmetric_cosine:
            sumOfSquared_to_1_minus_alpha = np.power(sumOfSquared, 2 * (1 - self.asymmetric_alpha))
            sumOfSquared_to_alpha = np.power(sumOfSquared, 2 * self.asymmetric_alpha)


        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col




        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:


            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block-start_col_block



            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            if self.use_row_weights:
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities
                this_block_weights = self.dataMatrix.T.dot(item_data)



            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:,col_index_in_block]


                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:

                    if self.asymmetric_cosine:
                        denominator = sumOfSquared_to_alpha[columnIndex] * sumOfSquared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)


                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + \
                                  (sumOfSquared[columnIndex] - this_column_weights)*self.tversky_alpha + \
                                  (sumOfSquared - this_column_weights)*self.tversky_beta + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights/self.shrink


                #this_column_weights = this_column_weights.toarray().ravel()

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK-1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)


            # Add previous block size
            processedItems += this_block_size


            if time.time() - start_time_print_batch >= 30 or end_col_block==end_col_local:
                columnPerSec = processedItems / (time.time() - start_time + 1e-9)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec, (time.time() - start_time)/ 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()


            start_col_block += block_size

        # End while on columns

        W_sparse = sps.csr_matrix((values, (rows, cols)),
                                  shape=(self.n_columns, self.n_columns),
                                  dtype=np.float32)


        return W_sparse
    
class Compute_Similarity:


    def __init__(self, dataMatrix, use_implementation = "density", similarity = None, **args):
        """
        Interface object that will call the appropriate similarity implementation
        :param dataMatrix:
        :param use_implementation:      "density" will choose the most efficient implementation automatically
                                        "cython" will use the cython implementation, if available. Most efficient for sparse matrix
                                        "python" will use the python implementation. Most efficent for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        """

        assert np.all(np.isfinite(dataMatrix.data)), \
            "Compute_Similarity: Data matrix contains {} non finite values".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        self.dense = False

        if similarity == "euclidean":
            # This is only available here
            self.compute_similarity_object = Compute_Similarity_Euclidean(dataMatrix, **args)

        else:

            assert not (dataMatrix.shape[0] == 1 and dataMatrix.nnz == dataMatrix.shape[1]),\
                "Compute_Similarity: data has only 1 feature (shape: {}) with dense values," \
                " vector and set based similarities are not defined on 1-dimensional dense data," \
                " use Euclidean similarity instead.".format(dataMatrix.shape)

            if similarity is not None:
                args["similarity"] = similarity


            if use_implementation == "density":

                if isinstance(dataMatrix, np.ndarray):
                    self.dense = True

                elif isinstance(dataMatrix, sps.spmatrix):
                    shape = dataMatrix.shape

                    num_cells = shape[0]*shape[1]

                    sparsity = dataMatrix.nnz/num_cells

                    self.dense = sparsity > 0.5

                else:
                    print("Compute_Similarity: matrix type not recognized, calling default...")
                    use_implementation = "python"

                if self.dense:
                    print("Compute_Similarity: detected dense matrix")
                    use_implementation = "python"
                else:
                    use_implementation = "cython"





            if use_implementation == "cython":

                try:
                    from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
                    self.compute_similarity_object = Compute_Similarity_Cython(dataMatrix, **args)

                except ImportError:
                    print("Unable to load Cython Compute_Similarity, reverting to Python")
                    self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)


            elif use_implementation == "python":
                self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)

            else:

                raise  ValueError("Compute_Similarity: value for argument 'use_implementation' not recognized")





    def compute_similarity(self,  **args):

        return self.compute_similarity_object.compute_similarity(**args)
    
class BaseRecommender(object):
    """Abstract BaseRecommender"""

    RECOMMENDER_NAME = "Recommender_Base_Class"

    def __init__(self, URM_train, verbose=True):

        super(BaseRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self.n_users, self.n_items = self.URM_train.shape
        self.verbose = verbose

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold users.".format(
                self._cold_user_mask.sum(), self._cold_user_mask.sum()/self.n_users*100))


        self._cold_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) == 0

        if self._cold_item_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold items.".format(
                self._cold_item_mask.sum(), self._cold_item_mask.sum()/self.n_items*100))


    def _get_cold_user_mask(self):
        return self._cold_user_mask

    def _get_cold_item_mask(self):
        return self._cold_item_mask


    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.RECOMMENDER_NAME, string))

    def fit(self):
        pass

    def get_URM_train(self):
        return self.URM_train.copy()

    def set_items_to_ignore(self, items_to_ignore):
        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):
        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                     COMPUTE AND FILTER RECOMMENDATION LIST                          ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch


    def _remove_custom_items_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch


    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        :param user_id_array:       array containing the user indices whose recommendations need to be computed
        :param items_to_compute:    array containing the items whose scores are to be computed.
                                        If None, all items are computed, otherwise discarded items will have as score -np.inf
        :return:                    array (len(user_id_array), n_items) with the score.
        """
        raise NotImplementedError("BaseRecommender: compute_item_score not assigned for current recommender, unable to compute prediction scores")


    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)


        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list



    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################



    def save_model(self, folder_path, file_name = None):
        raise NotImplementedError("BaseRecommender: save_model not implemented")




    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        self._print("Loading complete")

class BaseMatrixFactorizationRecommender(BaseRecommender):
    """
    This class refers to a BaseRecommender KNN which uses matrix factorization,
    it provides functions to compute item's score as well as a function to save the W_matrix
    The prediction for cold users will always be -inf for ALL items
    """

    def __init__(self, URM_train, verbose=True):
        super(BaseMatrixFactorizationRecommender, self).__init__(URM_train, verbose=verbose)

        self.use_bias = False




    #########################################################################################################
    ##########                                                                                     ##########
    ##########                               COMPUTE ITEM SCORES                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors
        The prediction for cold users will always be -inf for ALL items
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array),\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array], self.ITEM_factors[items_to_compute,:].T)

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)


        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        return item_scores


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"USER_factors": self.USER_factors,
                              "ITEM_factors": self.ITEM_factors,
                              "use_bias": self.use_bias,
                            }

        if self.use_bias:
            data_dict_to_save["ITEM_bias"] = self.ITEM_bias
            data_dict_to_save["USER_bias"] = self.USER_bias
            data_dict_to_save["GLOBAL_bias"] = self.GLOBAL_bias

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        self._print("Saving complete") 
        
class BaseSimilarityMatrixRecommender(BaseRecommender):
    """
    This class refers to a BaseRecommender KNN which uses a similarity matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """

    def __init__(self, URM_train, verbose=True):
        super(BaseSimilarityMatrixRecommender, self).__init__(URM_train, verbose = verbose)

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False



    def _check_format(self):

        if not self._URM_train_format_checked:

            if self.URM_train.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("URM_train", "csr"))

            self._URM_train_format_checked = True

        if not self._W_sparse_format_checked:

            if self.W_sparse.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("W_sparse", "csr"))

            self._W_sparse_format_checked = True




    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"W_sparse": self.W_sparse}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")



    #########################################################################################################
    ##########                                                                                     ##########
    ##########                               COMPUTE ITEM SCORES                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################


class BaseItemSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()

        return item_scores



def okapi_BM_25(dataMatrix, K1=1.2, B=0.75):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :param K1:
    :param B:
    :return:
    """

    assert B>0 and B<1, "okapi_BM_25: B must be in (0,1)"
    assert K1>0,        "okapi_BM_25: K1 must be > 0"

    assert np.all(np.isfinite(dataMatrix.data)), \
        "okapi_BM_25: Data matrix contains {} non finite values".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))



    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)

    dataMatrix = sps.coo_matrix(dataMatrix)

    N = float(dataMatrix.shape[0])
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # calculate length_norm per document
    row_sums = np.ravel(dataMatrix.sum(axis=1))

    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    denominator = K1 * length_norm[dataMatrix.row] + dataMatrix.data
    denominator[denominator == 0.0] += 1e-9

    dataMatrix.data = dataMatrix.data * (K1 + 1.0) / denominator * idf[dataMatrix.col]

    return dataMatrix.tocsr()

class Incremental_Training_Early_Stopping(object):
    """
    This class provides a function which trains a model applying early stopping
    The term "incremental" refers to the model that is updated at every epoch
    The term "best" refers to the incremental model which corresponded to the best validation score
    The object must implement the following methods:
    _run_epoch(self, num_epoch)                 : trains the model for one epoch (e.g. calling another object implementing the training cython, pyTorch...)
    _prepare_model_for_validation(self)         : ensures the recommender being trained can compute the predictions needed for the validation step
    _update_best_model(self)                    : updates the best model with the current incremental one
    _train_with_early_stopping(.)               : Function that executes the training, validation and early stopping by using the previously implemented functions
    """

    def __init__(self):
        super(Incremental_Training_Early_Stopping, self).__init__()


    def get_early_stopping_final_epochs_dict(self):
        """
        This function returns a dictionary to be used as optimal parameters in the .fit() function
        It provides the flexibility to deal with multiple early-stopping in a single algorithm
        e.g. in NeuMF there are three model components each with its own optimal number of epochs
        the return dict would be {"epochs": epochs_best_neumf, "epochs_gmf": epochs_best_gmf, "epochs_mlp": epochs_best_mlp}
        :return:
        """

        return {"epochs": self.epochs_best}


    def _run_epoch(self, num_epoch):
        """
        This function should run a single epoch on the object you train. This may either involve calling a function to do an epoch
        on a Cython object or a loop on the data points directly in python
        :param num_epoch:
        :return:
        """
        raise NotImplementedError()


    def _prepare_model_for_validation(self):
        """
        This function is executed before the evaluation of the current model
        It should ensure the current object "self" can be passed to the evaluator object
        E.G. if the epoch is done via Cython or PyTorch, this function should get the new parameter values from
        the cython or pytorch objects into the self. pyhon object
        :return:
        """
        raise NotImplementedError()


    def _update_best_model(self):
        """
        This function is called when the incremental model is found to have better validation score than the current best one
        So the current best model should be replaced by the current incremental one.
        Important, remember to clone the objects and NOT to create a pointer-reference, otherwise the best solution will be altered
        by the next epoch
        :return:
        """
        raise NotImplementedError()



    def _train_with_early_stopping(self, epochs_max, epochs_min = 0,
                                   validation_every_n = None, stop_on_validation = False,
                                   validation_metric = None, lower_validations_allowed = None, evaluator_object = None,
                                   algorithm_name = "Incremental_Training_Early_Stopping"):
        """
        :param epochs_max:                  max number of epochs the training will last
        :param epochs_min:                  min number of epochs the training will last
        :param validation_every_n:          number of epochs after which the model will be evaluated and a best_model selected
        :param stop_on_validation:          [True/False] whether to stop the training before the max number of epochs
        :param validation_metric:           which metric to use when selecting the best model, higher values are better
        :param lower_validations_allowed:    number of contiguous validation steps required for the tranining to early-stop
        :param evaluator_object:            evaluator instance used to compute the validation metrics.
                                                If multiple cutoffs are available, the first one is used
        :param algorithm_name:              name of the algorithm to be displayed in the output updates
        :return: -
        Supported uses:
        - Train for max number of epochs with no validation nor early stopping:
            _train_with_early_stopping(epochs_max = 100,
                                        evaluator_object = None
                                        epochs_min,                 not used
                                        validation_every_n,         not used
                                        stop_on_validation,         not used
                                        validation_metric,          not used
                                        lower_validations_allowed,   not used
                                        )
        - Train for max number of epochs with validation but NOT early stopping:
            _train_with_early_stopping(epochs_max = 100,
                                        evaluator_object = evaluator
                                        stop_on_validation = False
                                        validation_every_n = int value
                                        validation_metric = metric name string
                                        epochs_min,                 not used
                                        lower_validations_allowed,   not used
                                        )
        - Train for max number of epochs with validation AND early stopping:
            _train_with_early_stopping(epochs_max = 100,
                                        epochs_min = int value
                                        evaluator_object = evaluator
                                        stop_on_validation = True
                                        validation_every_n = int value
                                        validation_metric = metric name string
                                        lower_validations_allowed = int value
                                        )
        """

        assert epochs_max >= 0, "{}: Number of epochs_max must be >= 0, passed was {}".format(algorithm_name, epochs_max)
        assert epochs_min >= 0, "{}: Number of epochs_min must be >= 0, passed was {}".format(algorithm_name, epochs_min)
        assert epochs_min <= epochs_max, "{}: epochs_min must be <= epochs_max, passed are epochs_min {}, epochs_max {}".format(algorithm_name, epochs_min, epochs_max)

        # Train for max number of epochs with no validation nor early stopping
        # OR Train for max number of epochs with validation but NOT early stopping
        # OR Train for max number of epochs with validation AND early stopping
        assert evaluator_object is None or\
               (evaluator_object is not None and not stop_on_validation and validation_every_n is not None and validation_metric is not None) or\
               (evaluator_object is not None and stop_on_validation and validation_every_n is not None and validation_metric is not None and lower_validations_allowed is not None),\
            "{}: Inconsistent parameters passed, please check the supported uses".format(algorithm_name)




        start_time = time.time()

        self.best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.epochs_best = 0

        epochs_current = 0

        while epochs_current < epochs_max and not convergence:

            self._run_epoch(epochs_current)

            # If no validation required, always keep the latest
            if evaluator_object is None:

                self.epochs_best = epochs_current

            # Determine whether a validaton step is required
            elif (epochs_current + 1) % validation_every_n == 0:

                print("{}: Validation begins...".format(algorithm_name))

                self._prepare_model_for_validation()

                # If the evaluator validation has multiple cutoffs, choose the first one
                results_run, results_run_string = evaluator_object.evaluateRecommender(self)
                current_metric_value = results_run.iloc[0][validation_metric]

                print("{}: {}".format(algorithm_name, results_run_string))

                # Update optimal model
                if not np.isfinite(current_metric_value):
                    if isinstance(self, BaseTempFolder):
                        # If the recommender uses BaseTempFolder, clean the temp folder
                        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

                    assert False, "{}: metric value is not a finite number, terminating!".format(self.RECOMMENDER_NAME)


                if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                    print("{}: New best model found! Updating.".format(algorithm_name))
                    self.best_validation_metric = current_metric_value
                    self._update_best_model()

                    self.epochs_best = epochs_current +1
                    lower_validatons_count = 0

                else:
                    lower_validatons_count += 1


                if stop_on_validation and lower_validatons_count >= lower_validations_allowed and epochs_current >= epochs_min:
                    convergence = True

                    elapsed_time = time.time() - start_time
                    new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                    print("{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                        algorithm_name, epochs_current+1, validation_metric, self.epochs_best, self.best_validation_metric, new_time_value, new_time_unit))


            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            print("{}: Epoch {} of {}. Elapsed time {:.2f} {}".format(
                algorithm_name, epochs_current+1, epochs_max, new_time_value, new_time_unit))

            epochs_current += 1

            sys.stdout.flush()
            sys.stderr.flush()

        # If no validation required, keep the latest
        if evaluator_object is None:

            self._prepare_model_for_validation()
            self._update_best_model()


        # Stop when max epochs reached and not early-stopping
        if not convergence:
            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            if evaluator_object is not None and self.best_validation_metric is not None:
                print("{}: Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                    algorithm_name, epochs_current, validation_metric, self.epochs_best, self.best_validation_metric, new_time_value, new_time_unit))
            else:
                print("{}: Terminating at epoch {}. Elapsed time {:.2f} {}".format(
                    algorithm_name, epochs_current, new_time_value, new_time_unit))
                


class BaseTempFolder(object):

    def __init__(self):
        super(BaseTempFolder, self).__init__()

        self.DEFAULT_TEMP_FILE_FOLDER = './result_experiments/__Temp_{}_{}/'.format(self.RECOMMENDER_NAME, os.getpid())


    def _get_unique_temp_folder(self, input_temp_file_folder = None):

        if input_temp_file_folder is None:
            print("{}: Using default Temp folder '{}'".format(self.RECOMMENDER_NAME, self.DEFAULT_TEMP_FILE_FOLDER))
            self._use_default_temp_folder = True
            output_temp_file_folder = get_unique_temp_folder(self.DEFAULT_TEMP_FILE_FOLDER)
        else:
            print("{}: Using Temp folder '{}'".format(self.RECOMMENDER_NAME, input_temp_file_folder))
            self._use_default_temp_folder = False
            output_temp_file_folder = get_unique_temp_folder(input_temp_file_folder)

        if not os.path.isdir(output_temp_file_folder):
            os.makedirs(output_temp_file_folder)


        return output_temp_file_folder



    def _clean_temp_folder(self, temp_file_folder):
        """
        Clean temporary folder only if the default one
        :return:
        """

        if  self._use_default_temp_folder:
            print("{}: Cleaning temporary files from '{}'".format(self.RECOMMENDER_NAME, temp_file_folder))
            shutil.rmtree(temp_file_folder, ignore_errors=True)

        else:
            print("{}: Maintaining temporary files due to a custom temp folder being selected".format(self.RECOMMENDER_NAME))                
                


class IALSRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """
    Binary/Implicit Alternating Least Squares (IALS)
    See:
    Y. Hu, Y. Koren and C. Volinsky, Collaborative filtering for implicit feedback datasets, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf
    R. Pan et al., One-class collaborative filtering, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf
    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    """

    RECOMMENDER_NAME = "IALSRecommender"

    AVAILABLE_CONFIDENCE_SCALING = ["linear", "log"]


    def fit(self, epochs = 300,
            num_factors = 20,
            confidence_scaling = "linear",
            alpha = 1.0,
            epsilon = 1.0,
            reg = 1e-3,
            init_mean=0.0,
            init_std=0.1,
            **earlystopping_kwargs):
        """
        :param epochs:
        :param num_factors:
        :param confidence_scaling: supported scaling modes for the observed values: 'linear' or 'log'
        :param alpha: Confidence weight, confidence c = 1 + alpha*r where r is the observed "rating".
        :param reg: Regularization constant.
        :param epsilon: epsilon used in log scaling only
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :return:
        """

        if confidence_scaling not in self.AVAILABLE_CONFIDENCE_SCALING:
           raise ValueError("Value for 'confidence_scaling' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_CONFIDENCE_SCALING, confidence_scaling))


        self.num_factors = num_factors
        self.alpha = alpha
        self.epsilon = epsilon
        self.reg = reg

        self.USER_factors = self._init_factors(self.n_users, False)  # don't need values, will compute them
        self.ITEM_factors = self._init_factors(self.n_items)


        self._build_confidence_matrix(confidence_scaling)


        warm_user_mask = np.ediff1d(self.URM_train.indptr) > 0
        warm_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) > 0

        self.warm_users = np.arange(0, self.n_users, dtype=np.int32)[warm_user_mask]
        self.warm_items = np.arange(0, self.n_items, dtype=np.int32)[warm_item_mask]

        self.regularization_diagonal = np.diag(self.reg * np.ones(self.num_factors))

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best




    def _build_confidence_matrix(self, confidence_scaling):

        if confidence_scaling == 'linear':
            self.C = self._linear_scaling_confidence()
        else:
            self.C = self._log_scaling_confidence()

        self.C_csc= check_matrix(self.C.copy(), format="csc", dtype = np.float32)




    def _linear_scaling_confidence(self):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha*C.data

        return C

    def _log_scaling_confidence(self):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)

        return C




    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()


    def _run_epoch(self, num_epoch):

        # fit user factors
        # VV = n_factors x n_factors
        VV = self.ITEM_factors.T.dot(self.ITEM_factors)

        for user_id in self.warm_users:
            # get (positive i.e. non-zero scored) items for user

            start_pos = self.C.indptr[user_id]
            end_pos = self.C.indptr[user_id + 1]

            user_profile = self.C.indices[start_pos:end_pos]
            user_confidence = self.C.data[start_pos:end_pos]

            self.USER_factors[user_id, :] = self._update_row(user_profile, user_confidence, self.ITEM_factors, VV)

        # fit item factors
        # UU = n_factors x n_factors
        UU = self.USER_factors.T.dot(self.USER_factors)

        for item_id in self.warm_items:

            start_pos = self.C_csc.indptr[item_id]
            end_pos = self.C_csc.indptr[item_id + 1]

            item_profile = self.C_csc.indices[start_pos:end_pos]
            item_confidence = self.C_csc.data[start_pos:end_pos]

            self.ITEM_factors[item_id, :] = self._update_row(item_profile, item_confidence, self.USER_factors, UU)



    def _update_row(self, interaction_profile, interaction_confidence, Y, YtY):
        """
        Update latent factors for a single user or item.
        Y = |n_interactions|x|n_factors|
        YtY =   |n_factors|x|n_factors|
        """

        # Latent factors ony of item/users for which an interaction exists in the interaction profile
        Y_interactions = Y[interaction_profile, :]

        # Following the notation of the original paper we report the update rule for the Item factors (User factors are identical):
        # Y are the item factors |n_items|x|n_factors|
        # Cu is a diagonal matrix |n_interactions|x|n_interactions| with the user confidence for the observed items
        # p(u) is a boolean vectors indexing only observed items. Here it will disappear as we already extract only the observed latent factors
        #       however, it will have an impact in the dimensions of the matrix, since it transforms Cu from a diagonal matrix to a row vector of 1 row and |n_interactions| columns
        # (Yt*Cu*Y + reg*I)^-1 * Yt*Cu*profile
        # which can be decomposed as
        # (YtY + Yt*(Cu-I)*Y + reg*I)^-1 * Yt*Cu*p(u)

        # A = (|n_interactions|x|n_factors|) dot (|n_interactions|x|n_interactions| ) dot (|n_interactions|x|n_factors| )
        #   = |n_factors|x|n_factors|
        # A_slow = Y_interactions.T.dot(np.diag(interaction_confidence - 1)).dot(Y_interactions)

        # if v = diag(|n_interactions|) and k = |n_interactions|x|n_factors|
        # computing np.diag(v).dot(k) will be SLOW
        # we use an equivalent formulation (v * k.T).T which is much faster
        A = Y_interactions.T.dot(((interaction_confidence - 1) * Y_interactions.T).T)

        B = YtY + A + self.regularization_diagonal

        return np.dot(np.linalg.inv(B), Y_interactions.T.dot(interaction_confidence))


    def _init_factors(self, num_factors, assign_values=True):

        if assign_values:
            return self.num_factors**-0.5*np.random.random_sample((num_factors, self.num_factors))

        else:
            return np.empty((num_factors, self.num_factors))


class BaseUserSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_weights_array.dot(self.URM_train).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_train).toarray()

        return item_scores


def TF_IDF(dataMatrix):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :return:
    """

    assert np.all(np.isfinite(dataMatrix.data)), \
        "TF_IDF: Data matrix contains {} non finite values.".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

    assert np.all(dataMatrix.data >= 0.0),\
        "TF_IDF: Data matrix contains {} negative values, computing the square root is not possible.".format(np.sum(dataMatrix.data < 0.0))

    # TFIDF each row of a sparse amtrix
    dataMatrix = sps.coo_matrix(dataMatrix)
    N = float(dataMatrix.shape[0])

    # calculate IDF
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # apply TF-IDF adjustment
    dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]

    return dataMatrix.tocsr()
#############################################################################################################################################
"EASE"
###############################################################################################################################################
class EASE_R_Recommender(BaseItemSimilarityMatrixRecommender):
    """ EASE_R_Recommender
        https://arxiv.org/pdf/1905.03375.pdf
     @article{steck2019embarrassingly,
      title={Embarrassingly Shallow Autoencoders for Sparse Data},
      author={Steck, Harald},
      journal={arXiv preprint arXiv:1905.03375},
      year={2019}
    }
    """

    RECOMMENDER_NAME = "EASE_R_Recommender"


    def __init__(self, URM_train, sparse_threshold_quota = None):
        super(EASE_R_Recommender, self).__init__(URM_train)
        self.sparse_threshold_quota = sparse_threshold_quota

    def fit(self, topK=None, l2_norm = 1e3, normalize_matrix = False, verbose = True):

        self.verbose = verbose

        start_time = time.time()
        self._print("Fitting model... ")

        if normalize_matrix:
            # Normalize rows and then columns
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)


        # Grahm matrix is X X^t, compute dot product
        similarity = Compute_Similarity(self.URM_train, shrink=0, topK=self.URM_train.shape[1], normalize=False, similarity = "cosine")
        grahm_matrix = similarity.compute_similarity().toarray()

        diag_indices = np.diag_indices(grahm_matrix.shape[0])

        grahm_matrix[diag_indices] += l2_norm

        P = np.linalg.inv(grahm_matrix)

        B = P / (-np.diag(P))

        B[diag_indices] = 0.0


        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Fitting model... done in {:.2f} {}".format( new_time_value, new_time_unit))

        # Check if the matrix should be saved in a sparse or dense format
        # The matrix is sparse, regardless of the presence of the topK, if nonzero cells are less than sparse_threshold_quota %
        if topK is not None:
            B = similarityMatrixTopK(B, k = topK, verbose = False)


        if self._is_content_sparse_check(B):
            self._print("Detected model matrix to be sparse, changing format.")
            self.W_sparse = check_matrix(B, format='csr', dtype=np.float32)

        else:
            self.W_sparse = check_matrix(B, format='npy', dtype=np.float32)
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense
        #
        #
        # if topK is None:
        #     self.W_sparse = B
        #     self._W_sparse_format_checked = True
        #     self._compute_item_score = self._compute_score_W_dense
        #
        # else:
        #     self.W_sparse = similarityMatrixTopK(B, k = topK, verbose = False)
        #     self.W_sparse = sps.csr_matrix(self.W_sparse)


    def _is_content_sparse_check(self, matrix):

        if self.sparse_threshold_quota is None:
            return False

        if sps.issparse(matrix):
            nonzero = matrix.nnz
        else:
            nonzero = np.count_nonzero(matrix)

        return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota



    def _compute_score_W_dense(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)#.toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores
        """
        user_profile_array = sp_i_train[users[:10]]
        item_scores = - np.ones((len(users[:10]), sp_i_train.shape[1]), dtype=np.float32)*np.inf
        #B = similarityMatrixTopK(sp_i_train, 10)
        W_sparse = check_matrix(sp_i_train, format='npy', dtype=np.float32)
        item_scores_all = user_profile_array.dot(W_sparse)#.toarray()
        """


    def load_model(self, folder_path, file_name = None):
        super(EASE_R_Recommender, self).load_model(folder_path, file_name = file_name)

        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense
######################################################################################################################################################
"SLIM"
######################################################################################################################################################

            
class SLIMElasticNetRecommender(BaseItemSimilarityMatrixRecommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train, verbose = True):
        super(SLIMElasticNetRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100):

        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        # Display ConvergenceWarning only once and not for every item it occurs
        warnings.simplefilter("once", category = ConvergenceWarning)

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)


            if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} {}. Items per second: {:.2f}".format(
                    currentItem+1,
                    100.0* float(currentItem+1)/n_items,
                    new_time_value,
                    new_time_unit,
                    float(currentItem)/elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)



"""
import multiprocessing
from multiprocessing import Pool
from functools import partial

class MultiThreadSLIM_ElasticNet(SLIMElasticNetRecommender, BaseItemSimilarityMatrixRecommender):

    def __init__(self, URM_train, verbose = True):

        super(MultiThreadSLIM_ElasticNet, self).__init__(URM_train, verbose = verbose)


    def _partial_fit(self, currentItem, X, topK):
        model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)
        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values
        #nnz_idx = model.coef_ > 0.0

        relevant_items_partition = (-model.coef_).argpartition(topK)[0:topK]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        notZerosMask = model.coef_[ranking] > 0.0
        ranking = ranking[notZerosMask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        return values, rows, cols

    def fit(self,l1_ratio=0.1,
                 positive_only=True,
                 topK = 100,
                 workers=multiprocessing.cpu_count()):

        assert l1_ratio>= 0 and l1_ratio<=1, "SLIM_ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        self.workers = workers

        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = self.URM_train.shape[1]
        # fit item's factors in parallel
        
        #oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, X=self.URM_train, topK=self.topK)
        
        #creo un pool con un certo numero di processi
        pool = Pool(processes=self.workers)
        
        #avvio il pool passando la funzione (con la parte fissa dell'input) 
        #e il rimanente parametro, variabile
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
"""        
######################################################################################################################################################
"item-KNN"
######################################################################################################################################################

class ItemKNNCFRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]



    def __init__(self, URM_train, verbose = True):
        super(ItemKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)


        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        
class UserKNNCFRecommender(BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, verbose = True):
        super(UserKNNCFRecommender, self).__init__(URM_train, verbose = verbose)



    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        
class RP3betaRecommender(BaseItemSimilarityMatrixRecommender):
    """ RP3beta recommender """

    RECOMMENDER_NAME = "RP3betaRecommender"

    def __init__(self, URM_train, verbose = True):
        super(RP3betaRecommender, self).__init__(URM_train, verbose = verbose)


    def __str__(self):
        return "RP3beta(alpha={}, beta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                        self.beta, self.min_rating, self.topK,
                                                                                        self.implicit, self.normalize_similarity)

    def fit(self, alpha=1., beta=0.6, min_rating=0, topK=100, implicit=False, normalize_similarity=True):

        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        
        # if X.dtype != np.float32:
        #     print("RP3beta fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM_train.shape[1])

        nonZeroMask = X_bool_sum!=0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu


        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1


            if time.time() - start_time_printBatch > 60:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)


        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        
class P3alphaRecommender(BaseItemSimilarityMatrixRecommender):
    """ P3alpha recommender """

    RECOMMENDER_NAME = "P3alphaRecommender"

    def __init__(self, URM_train, verbose = True):
        super(P3alphaRecommender, self).__init__(URM_train, verbose = verbose)


    def __str__(self):
        return "P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                            self.min_rating, self.topK, self.implicit,
                                                                            self.normalize_similarity)

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=False, normalize_similarity=False):

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity


        #
        # if X.dtype != np.float32:
        #     print("P3ALPHA fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1


            if time.time() - start_time_printBatch > 300:
                new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)

                self._print("Similarity column {} ({:4.1f}%), {:.2f} column/sec. Elapsed time {:.2f} {}".format(
                     current_block_start_row + block_dim,
                    100.0 * float( current_block_start_row + block_dim) / Pui.shape[1],
                    float( current_block_start_row + block_dim) / (time.time() - start_time),
                    new_time_value, new_time_unit))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))


        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        