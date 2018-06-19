using System;
using System.Collections.Generic;
using System.Linq;


namespace CSharpUnitTest
{
    // Implements the wrapper for XGBoost (https://github.com/dmlc/xgboost)

    /// <summary>
    /// XGBoost DMatrix (inspired from Python code)
    /// DMatrix is a internal data structure that used by XGBoost
    /// which is optimized for both memory efficiency and training speed.
    /// You can construct DMatrix from numpy.arrays
    /// </summary>
    public class DMatrix
    {
#if(DEBUG)
        // For debugging purposes.
        struct GcKeep
        {
            public unsafe float[] data;
            public unsafe uint[] groups;
            public unsafe float[] weights;
            public unsafe float[] labels;
            public unsafe /*size_t*/ ulong[] indptr;
            public unsafe uint[] indices;
        }
        private GcKeep _gcKeep;
#endif

        readonly string[] _featureNames;
        readonly string[] _featureTypes;
        readonly IntPtr _handle;

        public string[] FeatureNames { get { return _featureNames; } }
        public string[] FeatureTypes { get { return _featureTypes; } }
        public IntPtr Handle { get { return _handle; } }

        public DMatrix(float[] data, uint nrow, uint ncol, float[] labels = null, float missing = float.NaN,
                 float[] weights = null, uint[] groups = null,
                 IEnumerable<string> featureNames = null, IEnumerable<string> featureTypes = null)
        {
#if(DEBUG)
            _gcKeep = new GcKeep()
            {
                data = data,
                labels = labels,
                weights = weights,
                groups = groups
            };
#endif

            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixCreateFromMat(data, nrow, ncol, missing, ref _handle));

            if (labels != null)
                SetLabel(labels, nrow);
            if (weights != null)
                SetWeight(weights, nrow);
            if (groups != null)
                SetGroups(groups, nrow);

            _featureNames = featureNames == null ? null : featureNames.ToArray();
            _featureTypes = featureTypes == null ? null : featureTypes.ToArray();
        }

        private void AssertIsTrue(bool cond, string msg = null)
        {
            if (!cond)
                throw new Exception(msg);
        }

        public DMatrix(/*bst_ulong*/ uint numColumn, /*size_t*/ ulong[] indptr, uint[] indices, float[] data,
                 uint nrow, uint nelem, float[] labels = null,
                 float[] weights = null, uint[] groups = null,
                 IEnumerable<string> featureNames = null, IEnumerable<string> featureTypes = null)
        {
            AssertIsTrue(nrow + 1 == indptr.Length);
#if(DEBUG)
            _gcKeep = new GcKeep()
            {
                indptr = indptr,
                indices = indices,
                data = data,
                labels = labels,
                weights = weights,
                groups = groups
            };
#endif

            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixCreateFromCSREx(indptr,
                indices, data, (ulong)indptr.Length, nelem, numColumn, ref _handle));

            if (labels != null)
                SetLabel(labels, nrow);
            if (weights != null)
                SetWeight(weights, nrow);
            if (groups != null)
                SetGroups(groups, nrow);

            _featureNames = featureNames == null ? null : featureNames.ToArray();
            _featureTypes = featureTypes == null ? null : featureTypes.ToArray();

            AssertIsTrue(nrow == (int)GetNbRows());
            AssertIsTrue((int)GetNbCols() == numColumn);
        }

        ~DMatrix()
        {
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixFree(_handle));
        }

        public void SaveBinary(string name, int silent = 0)
        {
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixSaveBinary(_handle, name, silent));
        }

        public uint GetNbRows()
        {
            uint nb = 0;
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixNumRow(_handle, ref nb));
            return nb;
        }

        public uint GetNbCols()
        {
            uint nb = 0;
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixNumCol(_handle, ref nb));
            return nb;
        }

        public void SetLabel(float[] label, uint nrow)
        {
            SetFloatInfo("label", label, nrow);
        }

        public void SetWeight(float[] weight, uint nrow)
        {
            SetFloatInfo("weight", weight, nrow);
        }

        public void SetBaseMargin(float[] margin, uint nrow)
        {
            SetFloatInfo("base_margin", margin, nrow);
        }

        public void SetGroups(IEnumerable<uint> group, uint nrow)
        {
            var agroup = group.ToArray();
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixSetGroup(_handle, agroup, nrow));
        }

        private void SetFloatInfo(string field, IEnumerable<float> data, uint nrow)
        {
            float[] cont = data.ToArray();
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixSetFloatInfo(_handle, field, cont, nrow));
        }
    }
}

