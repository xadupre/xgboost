using System;
using System.Runtime.InteropServices;


namespace CSharpUnitTest
{
    public static class WrappedXGBoostInterface
    {
        public const string DllName = @"..\..\..\..\..\lib\xgboost.dll";

        #region helpers

        /// <summary>
        /// Checks if XGBoost has a pending error message. Raises an exception in that case.
        /// </summary>
        /// The class is public (and not internal) to allow unit testing.
        public static void Check(int res)
        {
            if (res != 0)
            {
                string mes = XGBGetLastError();
                throw new Exception(string.Format("XGBoost Error, code is {0}, error message is '{1}'.", res, mes));
            }
        }

        /// <summary>
        /// Implements an unsafe memcopy.
        /// </summary>
        public static void Copy(IntPtr src, int srcIndex,
                float[] dst, int count)
        {
            Marshal.Copy(src, dst, srcIndex, count);
        }

        /// <summary>
        /// Implements an unsafe memcopy.
        /// </summary>
        public static string CastString(IntPtr src)
        {
            return Marshal.PtrToStringAnsi(src);
        }

        #endregion

        #region API ERROR

        [DllImport(DllName)]
        public static extern string XGBGetLastError();

        #endregion

        #region API DMatrix

        [DllImport(DllName, EntryPoint = "XGDMatrixCreateFromMat", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixCreateFromMat(float[] data, /*bst_ulong*/ uint nrow, /*bst_ulong*/ uint ncol, float missing, ref IntPtr res);

        [DllImport(DllName, EntryPoint = "XGDMatrixCreateFromCSR", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixCreateFromCSR(/*bst_ulong*/ uint[] indptr, uint[] indices, float[] data,
            /*bst_ulong*/ uint nindptr, /*bst_ulong*/ uint nelem, ref IntPtr res);

        [DllImport(DllName, EntryPoint = "XGDMatrixFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixFree(IntPtr handle);

        [DllImport(DllName, EntryPoint = "XGDMatrixSetFloatInfo", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixSetFloatInfo(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)]string field, float[] array, /*bst_ulong*/ uint len);

        [DllImport(DllName, EntryPoint = "XGDMatrixSetGroup", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixSetGroup(IntPtr handle, uint[] groups, /*bst_ulong*/ uint length);

        [DllImport(DllName, EntryPoint = "XGDMatrixNumRow", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixNumRow(IntPtr handle, ref /*bst_ulong*/ uint res);

        [DllImport(DllName, EntryPoint = "XGDMatrixNumCol", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixNumCol(IntPtr handle, ref /*bst_ulong*/ uint res);

        [DllImport(DllName, EntryPoint = "XGDMatrixCreateFromFile", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixCreateFromFile([MarshalAs(UnmanagedType.LPStr)]string fname, int silent, out IntPtr handle);

        #endregion

        #region API Booster

        [DllImport(DllName, EntryPoint = "XGBoosterCreate", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterCreate(IntPtr[] handles, /*bst_ulong*/ uint len, ref IntPtr res);

        [DllImport(DllName, EntryPoint = "XGBoosterFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterFree(IntPtr handle);

        [DllImport(DllName, EntryPoint = "XGBoosterSetParam", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterSetParam(IntPtr handle,
                                                   [MarshalAs(UnmanagedType.LPStr)]string name,
                                                   [MarshalAs(UnmanagedType.LPStr)]string value);

        [DllImport(DllName, EntryPoint = "XGBoosterLoadModelFromBuffer", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterLoadModelFromBuffer(IntPtr handle, byte* buf, /*bst_ulong*/ uint len);

        [DllImport(DllName, EntryPoint = "XGBoosterGetModelRaw", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterGetModelRaw(IntPtr handle, ref /*bst_ulong*/ uint outLen, out byte* outDptr);

        [DllImport(DllName, EntryPoint = "XGBoosterGetNumInfoTest", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterGetNumInfoTest(IntPtr handle, IntPtr res, [MarshalAs(UnmanagedType.LPStr)]string nameStr);

        public static double XGBoosterGetNumInfo(IntPtr handle, string nameStr)
        {
            double[] info = new double[1];
            unsafe
            {
                fixed (double* pd = info)
                {
                    IntPtr ptr = (IntPtr)pd;
                    WrappedXGBoostInterface.XGBoosterGetNumInfoTest(handle, ptr, "NumTrees");
                }
            }
            return info[0];
        }

        #endregion

        #region API train

        [DllImport(DllName, EntryPoint = "XGBoosterUpdateOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterUpdateOneIter(IntPtr handle, int iter, IntPtr dtrain);

        [DllImport(DllName, EntryPoint = "XGBoosterBoostOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterBoostOneIter(IntPtr handle, IntPtr dtrain, float[] grad, float[] hess, /*bst_ulong*/ uint len);

        /// outResult is a char** pointer, ANSI encoding.
        [DllImport(DllName, EntryPoint = "XGBoosterEvalOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGBoosterEvalOneIter(IntPtr handle, int iter, IntPtr[] dmats,
                                 [In][MarshalAsAttribute(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] evnames,
            /*bst_ulong*/ uint len, out IntPtr outResult);

        #endregion

        #region API Predict

        /// This function returns a pointer on data XGBoost owns. It cannot be freed.
        [DllImport(DllName, EntryPoint = "XGBoosterPredict", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredict(IntPtr handle, IntPtr dmat,
                                                  int optionMask, uint ntreeLimit, ref /*bst_ulong*/ uint outLen,
                                                  ref /*float* */ IntPtr outResult);

        #endregion

        #region Custom API Predict

        [StructLayout(LayoutKind.Sequential)]
        public struct SparseEntry
        {
            public uint index;
            public float fvalue;
        };

        [DllImport(DllName, EntryPoint = "XGBoosterPredictNoInsideCache", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredictNoInsideCache(IntPtr handle, /* SparseEntry **/ IntPtr entries,
                                                    /*bst_ulong*/ uint nbEntries,
                                                  int optionMask, uint ntreeLimit, /*bst_ulong*/ uint outLen, /*bst_ulong*/ uint outLenBuffer,
                                                  float* outResult, float* predBuffer, uint* predCounter,
                                                  /*RegVec::FVec*/ IntPtr regVecFVec);

        [DllImport(DllName, EntryPoint = "XGBoosterPredictOutputSize", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredictOutputSize(IntPtr handle, /* SparseEntry **/ IntPtr entries,
            /*bst_ulong*/ uint nbEntries, int optionMask, uint ntreeLimit, ref /*bst_ulong*/ uint outLen, ref /*bst_ulong*/ uint outLenBuffer);

        [DllImport(DllName, EntryPoint = "XGBoosterLazyInit", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterLazyInit(IntPtr handle);

        [DllImport(DllName, EntryPoint = "XGBoosterCopyEntries", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterCopyEntries(IntPtr entries, /*bst_ulong*/ ref uint nbEntries, float* values, int* indices, float missing);

        [DllImport(DllName, EntryPoint = "XGBoosterPredictNoInsideCacheAllocate", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredictNoInsideCacheAllocate(int nb_features,
                                    /*RegTree::FVec* */ ref IntPtr regtreefvec);

        [DllImport(DllName, EntryPoint = "XGBoosterPredictNoInsideCacheFree", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterPredictNoInsideCacheFree(/*RegTree::FVec* */ IntPtr regtreefvec);


        [DllImport(DllName, EntryPoint = "XGDMatrixSaveBinary", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixSaveBinary(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)]string fname, int silent);

#if (!XGBOOST_RABIT)

        [DllImport(DllName, EntryPoint = "XGBoosterLoadRabitCheckpoint", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterLoadRabitCheckpoint(IntPtr handleBooster, ref int version);

        [DllImport(DllName, EntryPoint = "XGBoosterSaveRabitCheckpoint", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int XGBoosterSaveRabitCheckpoint(IntPtr handleBooster);

        [DllImport(DllName, EntryPoint = "XGDMatrixCreateFromCSREx", CallingConvention = CallingConvention.StdCall)]
        public static extern int XGDMatrixCreateFromCSREx(/*size_t* */ ulong[] indptr, uint[] indices, float[] data,
            /*size_t*/ ulong nindptr, /*size_t*/ ulong nelem, /*ulong*/ ulong num_col, ref IntPtr res);

#endif
        public class XGBoostTreeBuffer
        {
            private byte[] xgboostEntries;
            private float[] predBuffer;
            private uint[] predCounter;
            private IntPtr regTreeFVec;
            private int regTreeFVecLength;

            public byte[] XGBoostEntries { get { return xgboostEntries; } }
            public float[] PredBuffer { get { return predBuffer; } }
            public uint[] PredCounter { get { return predCounter; } }
            public IntPtr RegTreeFVec { get { return regTreeFVec; } }

            static public XGBoostTreeBuffer CreateInternalBuffer()
            {
                return new XGBoostTreeBuffer();
            }

            public XGBoostTreeBuffer()
            {
                xgboostEntries = null;
                predBuffer = null;
                predCounter = null;
                regTreeFVec = (IntPtr)0;
                regTreeFVecLength = 0;
            }

            ~XGBoostTreeBuffer()
            {
                if (regTreeFVec != (IntPtr)0)
                    XGBoosterPredictNoInsideCacheFree(regTreeFVec);
                regTreeFVec = (IntPtr)0;
                regTreeFVecLength = 0;
            }

            public void ResizeEntries(uint nb, uint nbFeatures)
            {
                uint xgboostEntriesSize = nb * (sizeof(float) + sizeof(uint));
                if (xgboostEntries == null || xgboostEntries.Length < xgboostEntriesSize ||
                    xgboostEntriesSize > xgboostEntries.Length * 2)
                    xgboostEntries = new byte[xgboostEntriesSize];
                if (regTreeFVec == (IntPtr)0 || regTreeFVecLength < nbFeatures || nbFeatures > regTreeFVecLength * 2)
                {
                    if (regTreeFVec != (IntPtr)0)
                        XGBoosterPredictNoInsideCacheFree(regTreeFVec);
                    WrappedXGBoostInterface.Check(XGBoosterPredictNoInsideCacheAllocate((int)nbFeatures, ref regTreeFVec));
                    regTreeFVecLength = (int)nbFeatures;
                }
            }

            public void ResizeOutputs(uint length, uint lengthBuffer, ref float[] predictedValues)
            {
                if (predictedValues == null || length > (ulong)predictedValues.Length)
                    predictedValues = new float[length];
                if (predBuffer == null || lengthBuffer > (ulong)predBuffer.Length)
                    predBuffer = new float[lengthBuffer];
                if (predCounter == null || lengthBuffer > (ulong)predCounter.Length)
                    predCounter = new uint[lengthBuffer];
            }

            public void ResizeOutputs(uint length, uint lengthBuffer, ref FloatVector predictedValues)
            {
                if (length > (ulong)predictedValues.Length)
                    predictedValues = new FloatVector((int)length, new float[length]);
                else
                    predictedValues = new FloatVector((int)length, predictedValues.Values);

                if (predBuffer == null || lengthBuffer > (ulong)predBuffer.Length)
                    predBuffer = new float[lengthBuffer];
                if (predCounter == null || lengthBuffer > (ulong)predCounter.Length)
                    predCounter = new uint[lengthBuffer];
            }
        }

        #endregion
    }
}
