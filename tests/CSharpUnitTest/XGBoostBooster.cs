using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;


namespace CSharpUnitTest
{
    public class Booster
    {
        private string[] _featureNames;
        private string[] _featureTypes;
        private readonly IntPtr _handle;
        private int _nbFeatures;

        public IntPtr Handle { get { return _handle; } }
        public int NbFeatures { get { return _nbFeatures; } }

        public static WrappedXGBoostInterface.XGBoostTreeBuffer CreateInternalBuffer()
        {
            return new WrappedXGBoostInterface.XGBoostTreeBuffer();
        }

        public Booster(Dictionary<string, string> parameters, DMatrix data, Booster continuousTraining)
        {
            _featureNames = null;
            _featureTypes = null;
            _nbFeatures = (int)data.GetNbCols();

            _handle = continuousTraining == null ? (IntPtr)null : continuousTraining.Handle;
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterCreate(new[] { data.Handle }, 1, ref _handle));

            if (parameters != null)
                SetParam(parameters);
        }

        public Booster(byte[] content, int nbFeatures)
        {
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterCreate(new IntPtr[] { }, 0, ref _handle));
            unsafe
            {
                fixed (byte* p = content)
                {
                    WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterLoadModelFromBuffer(_handle, p, (uint)content.Length));
                }
            }
            _nbFeatures = nbFeatures;
        }

        ~Booster()
        {
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterFree(_handle));
        }

        private void AssertIsTrue(bool cond, string msg = null)
        {
            if (!cond)
                throw new Exception(msg);
        }


        private void ValidateFeatures(DMatrix data)
        {
            if (_featureNames == null)
            {
                AssertIsTrue((data.FeatureNames == null || data.FeatureTypes == null) || (data.FeatureNames.Length == data.FeatureTypes.Length));
                _featureNames = data.FeatureNames;
                _featureTypes = data.FeatureTypes;
            }
            else
            {
                AssertIsTrue(data.FeatureNames.Length == data.FeatureTypes.Length, "Length mismatch.");
                AssertIsTrue(data.FeatureNames.Length == _featureNames.Length, "New data does not have the same number of feature names (continuous training).");
                AssertIsTrue(data.FeatureTypes.Length == _featureTypes.Length, "New data does not have the same number of feature types (continuous training).");
                for (int i = 0; i < _featureNames.Length; ++i)
                {
                    AssertIsTrue(_featureNames[i] == data.FeatureNames[i], "Feature names are different in Booster and DMatrix.");
                    AssertIsTrue(_featureTypes[i] == data.FeatureTypes[i], "Idem for types.");
                }
            }
        }

        public void SetParam(Dictionary<string, string> parameters)
        {
            foreach (var pair in parameters)
                WrappedXGBoostInterface.XGBoosterSetParam(_handle, pair.Key, pair.Value);
        }

        public delegate void FObjType(ref FloatVector h, DMatrix dmat, ref FloatVector grad, ref FloatVector hess);

        public void Update(DMatrix dtrain, int iteration,
                            ref FloatVector grad, ref FloatVector hess, ref FloatVector prediction,
                            FObjType fobj = null)
        {
            ValidateFeatures(dtrain);

            if (fobj == null)
            {
                WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(_handle, iteration,
                    dtrain.Handle));
            }
            else
            {
                PredictN(dtrain, ref prediction);
                fobj(ref prediction, dtrain, ref grad, ref hess);
                Boost(dtrain, ref grad, ref hess);
            }
        }

        private void Boost(DMatrix dtrain, ref FloatVector grad, ref FloatVector hess)
        {
            AssertIsTrue(grad.Length == hess.Length, string.Format("grad / hess length mismatch: {0} / {1}", grad.Length, hess.Length));
            ValidateFeatures(dtrain);
            AssertIsTrue(grad.IsDense, "grad");
            AssertIsTrue(hess.IsDense, "hess");
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterBoostOneIter(_handle, dtrain.Handle, grad.Values, hess.Values, (uint)grad.Length));
        }

        public void LazyInit()
        {
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterLazyInit(_handle));
        }

        /// <summary>
        /// Predict with data. Calls the official API of XGBoost library.
        /// The function is protected against concurrent calls.
        /// </summary>
        public void PredictN(DMatrix data, ref FloatVector predictedValues, bool outputMargin = true, int ntreeLimit = 0)
        {
            int optionMask = 0x00;
            if (outputMargin)
                optionMask |= 0x01;

            // REVIEW xadupre: see review in function PredictOneOff.

            ValidateFeatures(data);
            uint length = 0;
            IntPtr ppreds = (IntPtr)0;

            unsafe
            {
                // XGBoost uses OMP to parallelize the computation
                // of the output, each observation will be computed in a separate thread
                // and will use thread specific context.
                // Read https://blogs.msdn.microsoft.com/oldnewthing/20101122-00/?p=12233.
                // This function is called from multiple threads in C# for the evaluation with an iterator,
                // XGBoost parallelizes the computation for each evaluation (even if it is one in this case).
                // It chooses the number of thread with: nthread = omp_get_num_threads() (gbtree.cc) 
                // The lock nullifies the parallelization done by TLC. 
                // There is no parallelization done by XGBoost on one observation.
                // Without the lock, the program fails (null pointer or something similar).
                // This item is a request: https://github.com/dmlc/xgboost/issues/1449.
                // As a consequence, this function is only used during training to evaluate the model on a batch of observations.
                // The reason is XGBoost is using caches in many places assuming XGBoost
                // is called from one unique thread.
                // That explains this lock.
                // That function only relies on the offical API of XGBoost.
                lock (this)
                {
                    int t = WrappedXGBoostInterface.XGBoosterPredict(_handle, data.Handle,
                        optionMask, (uint)ntreeLimit,
                        ref length, ref ppreds);
                    WrappedXGBoostInterface.Check(t);
                }
                float* preds = (float*)ppreds;
                AssertIsTrue(0 < length && length < Int32.MaxValue);
                if (length > (ulong)predictedValues.Length)
                    predictedValues = new FloatVector((int)length, new float[length]);
                WrappedXGBoostInterface.Copy((IntPtr)preds, 0, predictedValues.Values, (int)length);
            }
        }

        /// <summary>
        /// Predict with data.
        /// This function uses a modified API which does not use caches.
        /// </summary>
        public void PredictOneOff(ref FloatVector vbuf, ref FloatVector predictedValues,
                            ref WrappedXGBoostInterface.XGBoostTreeBuffer internalBuffer,
                            bool outputMargin = true, int ntreeLimit = 0)
        {
            int optionMask = 0x00;
            if (outputMargin)
                optionMask |= 0x01;

            // XGBoost can produce an output per tree (pred_leaf=true)
            // When this option is on, the output will be a matrix of (nsample, ntrees)
            // with each record indicating the predicted leaf index of each sample in each tree.
            // Note that the leaf index of a tree is unique per tree, so you may find leaf 1
            // in both tree 1 and tree 0.
            // if (pred_leaf)
            //    option_mask |= 0x02;
            // This might be an interesting feature to implement.

            AssertIsTrue(internalBuffer != null);

            uint length = 0;
            uint lengthBuffer = 0;
            uint nb = (uint)vbuf.Count;

            // This function relies on a modified API. Instead of letting XGBoost handle its own caches,
            // the function calls XGBoosterPredictOutputSize to know what cache size is required.
            // TLC allocated the caches and gives them to XGBoost.
            // First, we allocated the cache for the features. Only then XGBoost
            // will be able to known the required cache size.
            internalBuffer.ResizeEntries(nb, (uint)vbuf.Length);

            unsafe
            {
                fixed (float* p = vbuf.Values)
                fixed (int* i = vbuf.Indices)
                fixed (byte* entries = internalBuffer.XGBoostEntries)
                {
                    WrappedXGBoostInterface.XGBoosterCopyEntries((IntPtr)entries, ref nb, p, vbuf.IsDense ? null : i, float.NaN);
                    WrappedXGBoostInterface.XGBoosterPredictOutputSize(_handle,
                        (IntPtr)entries, nb, optionMask, (uint)ntreeLimit, ref length, ref lengthBuffer);
                }
            }

            // Then we allocated the cache for the prediction.
            internalBuffer.ResizeOutputs(length, lengthBuffer, ref predictedValues);

            unsafe
            {
                fixed (byte* entries = internalBuffer.XGBoostEntries)
                fixed (float* ppreds = predictedValues.Values)
                fixed (float* ppredBuffer = internalBuffer.PredBuffer)
                fixed (uint* ppredCounter = internalBuffer.PredCounter)
                {
                    WrappedXGBoostInterface.XGBoosterPredictNoInsideCache(_handle,
                        (IntPtr)entries, nb, optionMask, (uint)ntreeLimit, length, lengthBuffer, ppreds, ppredBuffer, ppredCounter
                        , internalBuffer.RegTreeFVec
                        );
                }
            }
        }

        public void Predict(ref FloatVector features,
                            ref FloatVector predictedValues,
                            ref WrappedXGBoostInterface.XGBoostTreeBuffer internalBuffer,
                            bool outputMargin = true,
                            int ntreeLimit = 0)
        {
            PredictOneOff(ref features, ref predictedValues, ref internalBuffer, outputMargin, ntreeLimit);

#if (DEBUG && MORE_CHECKING)
            // This part checks that the function PredictOneOff which relies on a customized version 
            // of XGBoost produces the same result as the official API.
            // This makes the prediction terribly slow as the prediction are called twice
            // and the second call (PredictN) cannot be parallelized (lock protected).
            FloatVector check = new FloatVector();
            DMatrix data;
            if (features.IsDense)
                data = new DMatrix(features.Values, 1, (uint)features.Count);
            else
            {
                int nb = features.Count;
                var indptr = new ulong[] { 0, (uint)nb };
                var indices = new uint[nb];
                for (int i = 0; i < nb; ++i)
                    indices[i] = (uint)features.Indices[i];
                data = new DMatrix((uint)features.Length, indptr, indices, features.Values, 1, (uint)nb);
            }

            PredictN(data, ref check, outputMargin, ntreeLimit);
            if (check.Count != predictedValues.Count)
            {
                string message =
                    string.Format(
                        "Count={0} Length={1} IsDense={2}\nValues={3}\nIndices={4}\nCustom Ouput={5}\nOfficial API={6}",
                        features.Count, features.Length, features.IsDense,
                        features.Values == null
                            ? ""
                            : string.Join(", ", features.Values.Select(c => c.ToString()).ToArray()),
                        features.Indices == null
                            ? ""
                            : string.Join(", ", features.Indices.Select(c => c.ToString()).ToArray()),
                        predictedValues.Values == null
                            ? ""
                            : string.Join(", ", predictedValues.Values.Select(c => c.ToString()).ToArray()),
                        check.Values == null
                            ? ""
                            : string.Join(", ", check.Values.Select(c => c.ToString()).ToArray()));
                throw Contracts.Except("Mismatch between official API and custom API (dimension).\n" + message);
            }
            for (int i = 0; i < check.Count; ++i)
            {
                if (Math.Abs(check.Values[0] - predictedValues.Values[0]) > 1e-5)
                {
                    string message =
                        string.Format(
                            "Count={0} Length={1} IsDense={2}\nValues={3}\nIndices={4}\nCustom Ouput={5}\nOfficial API={6}",
                            features.Count, features.Length, features.IsDense,
                            features.Values == null
                                ? ""
                                : string.Join(", ", features.Values.Select(c => c.ToString()).ToArray()),
                            features.Indices == null
                                ? ""
                                : string.Join(", ", features.Indices.Select(c => c.ToString()).ToArray()),
                            predictedValues.Values == null
                                ? ""
                                : string.Join(", ", predictedValues.Values.Select(c => c.ToString()).ToArray()),
                            check.Values == null
                                ? ""
                                : string.Join(", ", check.Values.Select(c => c.ToString()).ToArray()));
                    PredictOneOff(ref features, ref predictedValues, ref internalBuffer, outputMargin, ntreeLimit);
                    message += string.Format("\nSecond computation\n{0}", predictedValues.Values == null
                        ? ""
                        : string.Join(", ", predictedValues.Values.Select(c => c.ToString()).ToArray()));
                    throw Contracts.Except("Mismatch between official API and custom API (output).\n" + message);
                }
            }
#endif
        }

        public string EvalSet(DMatrix[] dmats, string[] names, int iteration = 0)
        {
            IntPtr outResult;
            for (int i = 0; i < dmats.Length; ++i)
                ValidateFeatures(dmats[i]);
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterEvalOneIter(Handle, iteration,
                dmats.Select(c => c.Handle).ToArray(),
                names, (uint)dmats.Length, out outResult));

            return WrappedXGBoostInterface.CastString(outResult);
        }

        public byte[] SaveRaw()
        {
            unsafe
            {
                byte* buffer;
                uint size = 0;
                WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterGetModelRaw(_handle, ref size, out buffer));
                byte[] content = new byte[size];
                Marshal.Copy((IntPtr)buffer, content, 0, content.Length);
                return content;
            }
        }

        public int LoadRabitCheckpoint()
        {
            int version = 0;
#if (!XGBOOST_RABIT)
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterLoadRabitCheckpoint(_handle, ref version));
#endif
            return version;
        }

        public void SaveRabitCheckpoint()
        {
#if (!XGBOOST_RABIT)
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSaveRabitCheckpoint(_handle));
#endif
        }
    }
}

