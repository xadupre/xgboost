/////////////////////////////////////
// ADDITION TO GET ONE OFF PREDICTION
/////////////////////////////////////


// Replicates SparseBatch::Entry
typedef struct  {
  /*! \brief feature index */
  uint32_t index;
  /*! \brief feature value */
  float fvalue;
} SparseEntry;

/*!
 * \brief make prediction on one observation, XGBoost does not use internal caches.
 * This function can be called from different threads.
 * \param handle handle
 * \param entries pointer on the input vector
 * \param nb_entries dimension of the input vector
 * \param option_mask bit-mask of options taken in prediction, possible values
 *          0:normal prediction
 *          1:output margin instead of transformed value
 *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
 * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
 *    when the parameter is set to 0, we will use all the trees
 * \param XGBoosterPredictOutputSizeout_len size of buffer out_result
 * \param XGBoosterBufferOutputSizeout_len size of buffer pred_buffer and pred_counter
 * \param out_result allocated buffer for prediction (it must be allocated by the user which gets its size by calling function XGBoosterPredictOutputSize)
 * \param pred_buffer allocated internal buffer (it must be allocated by the user which gets its size by calling function XGBoosterPredictOutputSize)
 * \param pred_counter (it must be allocated by the user which gets its size by calling function XGBoosterPredictOutputSize)
 * \param regtreefvec (it must be allocated by the user, size = number of features)
 * \return 0 when success, -1 when failure happens
 */
 XGB_DLL int XGBoosterPredictNoInsideCache(BoosterHandle handle,
                                          const SparseEntry * entries,
                                          bst_ulong nb_entries,
                                          int option_mask,
                                          unsigned ntree_limit,
                                          bst_ulong XGBoosterPredictOutputSizeout_len,
                                          bst_ulong XGBoosterBufferOutputSizeout_len,
                                          const float *out_result,
                                          const float *pred_buffer,
                                          const unsigned *pred_counter,
										  const /*RegTree::FVec*/ void* regtreefvec);

/*!
* Allocate internal buffers for XGBoosterPredictNoInsideCache
* \param nb_features number of features
* \param regtreefvec output
* \return 0 when success, -1 when failure happens
*/
XGB_DLL int XGBoosterPredictNoInsideCacheAllocate(int nb_features, /*RegTree::FVec*/ void** regtreefvec);

/*!
* Free internal buffers for XGBoosterPredictNoInsideCache
* \param regtreefvec allocated buffer
* \return 0 when success, -1 when failure happens
*/
XGB_DLL int XGBoosterPredictNoInsideCacheFree(/*RegTree::FVec*/ void* regtreefvec);

/*!
 * \brief return the size of the cache XGBoost needs to compute the one off predictions.
 * \param handle handle
 * \param entries pointer on the input vector
 * \param nb_entries dimension of the input vector
 * \param option_mask bit-mask of options taken in prediction, possible values
 *          0:normal prediction
 *          1:output margin instead of transformed value
 *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
 * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
 *    when the parameter is set to 0, we will use all the trees
 * \param out_len size of buffer out_result in fucntion XGBoosterPredictNoInsideCache
 * \param out_len_buffer size of buffer pred_buffer and pred_counter in function XGBoosterPredictNoInsideCache
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictOutputSize(BoosterHandle handle,
                                       const SparseEntry * entries,
                                       bst_ulong nb_entries,
                                       int option_mask,
                                       unsigned ntree_limit,
                                       bst_ulong *out_len,
                                       bst_ulong *out_len_buffer);

/**
* Calls the lazy initialisation of a booster.
*/
XGB_DLL int XGBoosterLazyInit(BoosterHandle handle);

/** Copy a sparse vector (values, indices) into SparseEntry vector.
* \param entries pointer on the input vector
* \param nb_entries dimension of the input vector
* \param values values to copy
* \param indices indices to copy
* \param missing value of the missing value (every value equal to missing will be skipped)
*/
XGB_DLL int XGBoosterCopyEntries(SparseEntry * entries,
                                 bst_ulong * nb_entries,
                                 const float * values,
                                 const int * indices,
                                 float missing);

/*! \brief handle to a internal data holder. */
typedef void *ArrayVoidHandle;

/*!
* \brief return one numeric features about xgboost
* \param out handle to the result booster
* \param nameStr name of the request
* \param outNum numeric feature
* \return 0 when success, -1 when failure happens
*/
XGB_DLL int XGBoosterGetNumInfoTest(BoosterHandle handle, ArrayVoidHandle outNum, const char* nameStr);

//////////////////
// END OF ADDITION
//////////////////
