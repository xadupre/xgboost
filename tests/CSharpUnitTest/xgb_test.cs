using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;


namespace CSharpUnitTest
{
    public static class xgb_test
    {
        static private string PATH = @"data";

        static string GetDataDir(string name)
        {
            var location = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            return Path.Combine(location, "..", "..", "..", PATH, name.Replace("/", "\\"));
        }

        static private IEnumerable<float[]> ReadCsv(string filename, char sep = '\t')
        {
            using (StreamReader sr = new StreamReader(filename))
            {
                var line = sr.ReadLine();
                while (line != null)
                {
                    var pl = line.Trim(new char[] { '\r', '\n', '\t', ' ' }).Split(sep);
                    yield return pl.Select(c => System.Convert.ToSingle(c)).ToArray();
                    line = sr.ReadLine();
                }
            }
        }

        static private void PredictOneOff(IntPtr handle, ref float[] vbuf, ref float[] predictedValues,
                                            ref WrappedXGBoostInterface.XGBoostTreeBuffer internalBuffer,
                            bool outputMargin = true, int ntreeLimit = 0)
        {
            int optionMask = 0x00;
            if (outputMargin)
                optionMask |= 0x01;

            // REVIEW xadupre: XGBoost can produce an output per tree (pred_leaf=true)
            // When this option is on, the output will be a matrix of (nsample, ntrees)
            // with each record indicating the predicted leaf index of each sample in each tree.
            // Note that the leaf index of a tree is unique per tree, so you may find leaf 1
            // in both tree 1 and tree 0.
            // if (pred_leaf)
            //    option_mask |= 0x02;
            // This might be an interesting feature to implement.

            uint length = 0;
            uint lengthBuffer = 0;
            uint nb = (uint)vbuf.Length;
            internalBuffer.ResizeEntries(nb, nb);

            unsafe
            {
                fixed (float* p = vbuf)
                fixed (byte* entries = internalBuffer.XGBoostEntries)
                {
                    WrappedXGBoostInterface.XGBoosterCopyEntries((IntPtr)entries, ref nb, p, null, float.NaN);
                    WrappedXGBoostInterface.XGBoosterPredictOutputSize(handle,
                        (IntPtr)entries, nb, optionMask, (uint)ntreeLimit, ref length, ref lengthBuffer);
                }
            }

            // The output is dense.
            internalBuffer.ResizeOutputs(length, lengthBuffer, ref predictedValues);

            unsafe
            {
                fixed (byte* entries = internalBuffer.XGBoostEntries)
                fixed (float* ppreds = predictedValues)
                fixed (float* ppredBuffer = internalBuffer.PredBuffer)
                fixed (uint* ppredCounter = internalBuffer.PredCounter)
                {
                    WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterPredictNoInsideCache(handle,
                        (IntPtr)entries, nb, optionMask, (uint)ntreeLimit, length, lengthBuffer, ppreds, ppredBuffer,
                        ppredCounter, internalBuffer.RegTreeFVec));
                }
            }
        }

        static public void PredictN(IntPtr data, IntPtr handle, ref float[] predictedValues, ref object lockobj, bool outputMargin = true, int ntreeLimit = 0)
        {
            int optionMask = 0x00;
            if (outputMargin)
                optionMask |= 0x01;
            uint length = 0;
            IntPtr ppreds = (IntPtr)0;
            unsafe
            {
                lock (lockobj)
                {
                    int t = WrappedXGBoostInterface.XGBoosterPredict(handle, data,
                        optionMask, (uint)ntreeLimit,
                        ref length, ref ppreds);
                    WrappedXGBoostInterface.Check(t);
                }
                float* preds = (float*)ppreds;
                if (predictedValues == null || length > (ulong)predictedValues.Length)
                    predictedValues = new float[length];
                WrappedXGBoostInterface.Copy((IntPtr)preds, 0, predictedValues, (int)length);
            }
        }

        static public IntPtr CreateDMatrix(IEnumerable<float[]> elements, float[] groups = null)
        {
            List<float> data = new List<float>();
            List<float> labels = new List<float>();
            int nrow = 0;
            int ncol = 0;
            foreach (var row in elements)
            {
                data.AddRange(row.Skip(1));
                labels.Add(row[0]);
                ++nrow;
                ncol = row.Length - 1;
            }

            IntPtr dmat = (IntPtr)0;
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixCreateFromMat(data.ToArray(), (uint)nrow, (uint)ncol, float.NaN, ref dmat));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixSetFloatInfo(dmat, "label", labels.ToArray(), (uint)labels.Count));
            if (groups != null)
                WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGDMatrixSetFloatInfo(dmat, "group", groups.ToArray(), (uint)groups.Length));

            return dmat;
        }

        static public void XGBoostCompareEvaluationDartWholeFileTest()
        {
            var lockobj = new object();
            var data = GetDataDir("housing.txt");
            var model = GetDataDir("xgbreg-dart-0.6-2016-08-26.xgb");
            var bin = File.ReadAllBytes(model);
            IntPtr handle = (IntPtr)0;
            WrappedXGBoostInterface.XGBoosterCreate(null, 0, ref handle);
            WrappedXGBoostInterface.XGBoosterLazyInit(handle);

            var datamat = ReadCsv(data).ToList();
            var allElements_ = datamat.Select((c, i) => new { i = i, f = c.Skip(1).ToArray() }).ToList();
            var ac = allElements_.ToArray();
            for (int i = 0; i < 40; ++i)
                allElements_.AddRange(ac);
            var allElements = allElements_.Select((c, i) => new { i = i, f = c.f }).ToList();

            var c1s = new double[allElements.Count];
            var c2s = new double[allElements.Count];
            var c3s = new double[allElements.Count];
            var f1s = new float[allElements.Count][];
            var f2s = new float[allElements.Count][];
            var f3s = new float[allElements.Count][];
            int stop1 = 697;
            var sharedBuffer = WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();

            IntPtr dmat2 = CreateDMatrix(datamat);

            unsafe
            {
                fixed (byte* b = bin)
                    //WrappedXGBoostInterface.XGBoosterLoadModelFromBuffer(handle, b, (uint)bin.Length);
                    WrappedXGBoostInterface.XGBoosterCreate(new IntPtr[] { dmat2 }, 1, ref handle);

            }

            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "objective", "reg:linear"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "max_depth", "3"));
            // Unless we fix the seed for dart booster, predictions may be different.
            // WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "booster", "dart"));

            int it = 0;
            IntPtr outResult;
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(handle, it, dmat2));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(handle, it, dmat2));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterEvalOneIter(handle, it++, new IntPtr[] { dmat2 }, new string[] { "ok" }, 1, out outResult));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(handle, it, dmat2));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterEvalOneIter(handle, it++, new IntPtr[] { dmat2 }, new string[] { "ok" }, 1, out outResult));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(handle, it, dmat2));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterEvalOneIter(handle, it++, new IntPtr[] { dmat2 }, new string[] { "ok" }, 1, out outResult));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(handle, it, dmat2));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterEvalOneIter(handle, it++, new IntPtr[] { dmat2 }, new string[] { "ok" }, 1, out outResult));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(handle, it, dmat2));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterEvalOneIter(handle, it++, new IntPtr[] { dmat2 }, new string[] { "ok" }, 1, out outResult));
            WrappedXGBoostInterface.XGDMatrixFree(dmat2);

            foreach (var obj in allElements)
            {
                var features = obj.f;
                var b = DateTime.Now;
                PredictOneOff(handle, ref features, ref f3s[obj.i], ref sharedBuffer);
                c3s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                IntPtr dmat = (IntPtr)0;
                unsafe
                {
                    WrappedXGBoostInterface.XGDMatrixCreateFromMat(features, 1, (uint)features.Length, float.NaN,
                        ref dmat);
                }
                PredictN(dmat, handle, ref f1s[obj.i], ref lockobj);
            }

            int nbThread1 = 0;
            System.Threading.Tasks.Parallel.ForEach(allElements,
                () =>
                {
                    nbThread1 += 1;
                    return WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();
                },
                (obj, loop, buffer) =>
                {
                    var features = obj.f;
                    var b = DateTime.Now;

                    IntPtr dmat = (IntPtr)0;
                    unsafe
                    {
                        WrappedXGBoostInterface.XGDMatrixCreateFromMat(features, 1, (uint)features.Length, float.NaN,
                            ref dmat);
                    }
                    PredictN(dmat, handle, ref f1s[obj.i], ref lockobj);
                    unsafe
                    {
                        WrappedXGBoostInterface.XGDMatrixFree(dmat);
                    }

                    c1s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                    return buffer;
                },
                _ => { });

            int nbThread2 = 0;
            System.Threading.Tasks.Parallel.ForEach(allElements,
                () =>
                {
                    nbThread2 += 1;
                    return WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();
                },
                (obj, loop, buffer) =>
                {
                    // Custom API
                    var features = obj.f;
                    var b = DateTime.Now;
                    PredictOneOff(handle, ref features, ref f2s[obj.i], ref buffer);
                    c2s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                    return buffer;
                },
                _ => { });

            var issues = new List<int>();
            for (int i = 0; i < stop1; ++i)
            {
                if (f1s[i].Length != f2s[i].Length)
                {
                    issues.Add(i);
                    break;
                }
                var d = Math.Abs(f1s[i][0] - f2s[i][0]);
                if (d > 1e-5)
                    issues.Add(i);
            }

            unsafe
            {
                WrappedXGBoostInterface.XGBoosterFree(handle);
            }

            var s1 = c1s.Sum() / allElements.Count;
            var s2 = c2s.Sum() / allElements.Count;
            var s3 = c3s.Sum() / allElements.Count;
            if (s1 < s2 / 2)
                throw new Exception(string.Format("LONGER {0} < {1} th1={2}, th2={3}", s1, s2, nbThread1, nbThread2));

            if (issues.Count > 0)
                throw new Exception(string.Join(", ", issues.Select(i => i.ToString()).ToArray()));

            Console.WriteLine("nbTh={4} s1={0} s2={1} s3={3}, N={2}", s1, s2, f1s.Length, s3, nbThread2);
        }

        static public void XGBoostCompareEvaluationRankingWholeFileTest()
        {
            var lockobj = new object();
            var data = GetDataDir("housing.txt");
            var model = GetDataDir("xgbrank-0.6-2016-09-02.xgb");
            var bin = File.ReadAllBytes(model);
            IntPtr handle = (IntPtr)0;
            WrappedXGBoostInterface.XGBoosterCreate(null, 0, ref handle);

            var rand = new Random();
            var datamat = new float[2000][];
            for (int l = 0; l < datamat.Length; ++l)
            {
                datamat[l] = new float[47];
                for (int h = 0; h < datamat[0].Length; ++h)
                    datamat[l][h] = (float)rand.NextDouble();
            }
            var allElements_ = datamat.Select((c, i) => new { i = i, f = c.Skip(1).ToArray() }).ToList();
            var ac = allElements_.ToArray();
            for (int i = 0; i < 10; ++i)
                allElements_.AddRange(ac);
            var allElements = allElements_.Select((c, i) => new { i = i, f = c.f }).ToList();
            var groups = allElements.Select(c => 0f).ToArray();

            var c1s = new double[allElements.Count];
            var c2s = new double[allElements.Count];
            var c3s = new double[allElements.Count];
            var f1s = new float[allElements.Count][];
            var f2s = new float[allElements.Count][];
            var f3s = new float[allElements.Count][];
            int stop1 = Math.Min(allElements.Count, 300);
            var sharedBuffer = WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();

            IntPtr dmat2 = CreateDMatrix(datamat, groups);

            unsafe
            {
                fixed (byte* b = bin)
                {
                    WrappedXGBoostInterface.XGBoosterLoadModelFromBuffer(handle, b, (uint)bin.Length);
                    WrappedXGBoostInterface.XGBoosterLazyInit(handle);
                }
                //WrappedXGBoostInterface.XGBoosterCreate(new IntPtr[] { dmat2 }, 1, ref handle);
            }

            //WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "objective", "reg:ranking"));

            foreach (var obj in allElements)
            {
                var features = obj.f;
                var b = DateTime.Now;
                PredictOneOff(handle, ref features, ref f3s[obj.i], ref sharedBuffer);
                c3s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
            }

            System.Threading.Tasks.Parallel.ForEach(allElements.Take(stop1),
                (obj) =>
                {
                    var features = obj.f;
                    var b = DateTime.Now;

                    IntPtr dmat = (IntPtr)0;
                    unsafe
                    {
                        WrappedXGBoostInterface.XGDMatrixCreateFromMat(features, 1, (uint)features.Length, float.NaN,
                            ref dmat);
                    }
                    PredictN(dmat, handle, ref f1s[obj.i], ref lockobj);
                    unsafe
                    {
                        WrappedXGBoostInterface.XGDMatrixFree(dmat);
                    }

                    c1s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                });

            int nbThread = 0;
            System.Threading.Tasks.Parallel.ForEach(allElements, () =>
            {
                nbThread += 1;
                return WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();
            },
                (obj, loop, buffer) =>
                {
                    // Custom API
                    var features = obj.f;
                    var b = DateTime.Now;
                    PredictOneOff(handle, ref features, ref f2s[obj.i], ref buffer);
                    c2s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                    return buffer;
                },
                _ => { });

            var issues = new List<int>();
            for (int i = 0; i < stop1; ++i)
            {
                if (f1s[i].Length != f2s[i].Length)
                {
                    issues.Add(i);
                    break;
                }
                var d = Math.Abs(f1s[i][0] - f2s[i][0]);
                if (d > 1e-5)
                {
                    issues.Add(i);
                }
            }

            unsafe
            {
                WrappedXGBoostInterface.XGBoosterFree(handle);
            }

            var s1 = c1s.Sum() / stop1;
            var s2 = c2s.Sum() / allElements.Count;
            var s3 = c3s.Sum() / allElements.Count;

            if (issues.Count > 0)
                throw new Exception(string.Join(", ", issues.Select(i => i.ToString()).ToArray()));

            Console.WriteLine("nbTh={4} s1={0} s2={1} s3={3}, N={2}", s1, s2, f1s.Length, s3, nbThread);
        }

        static public void XGBoostCompareEvaluationBinaryWholeFileTest()
        {
            var lockobj = new object();
            var data = GetDataDir("housing.txt");
            var model = GetDataDir("xgbcl-0.6-2016-09-02.xgb");
            var bin = File.ReadAllBytes(model);
            IntPtr handle = (IntPtr)0;
            WrappedXGBoostInterface.XGBoosterCreate(null, 0, ref handle);

            var rand = new Random();
            var datamat = new float[2000][];
            for (int l = 0; l < datamat.Length; ++l)
            {
                datamat[l] = new float[10] { 0f, 3f, 1f, 1f, 1f, 2f, Single.NaN, 3f, 1f, 1f };
            }
            var allElements_ = datamat.Select((c, i) => new { i = i, f = c.Skip(1).ToArray() }).ToList();
            var ac = allElements_.ToArray();
            for (int i = 0; i < 10; ++i)
                allElements_.AddRange(ac);
            var allElements = allElements_.Select((c, i) => new { i = i, f = c.f }).ToList();
            var groups = allElements.Select(c => 0f).ToArray();

            var c1s = new double[allElements.Count];
            var c2s = new double[allElements.Count];
            var c3s = new double[allElements.Count];
            var f1s = new float[allElements.Count][];
            var f2s = new float[allElements.Count][];
            var f3s = new float[allElements.Count][];
            int stop1 = Math.Min(allElements.Count, 300);
            var sharedBuffer = WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();

            IntPtr dmat2 = CreateDMatrix(datamat, groups);

            unsafe
            {
                fixed (byte* b = bin)
                {
                    WrappedXGBoostInterface.XGBoosterLoadModelFromBuffer(handle, b, (uint)bin.Length);
                    WrappedXGBoostInterface.XGBoosterLazyInit(handle);
                }
                //WrappedXGBoostInterface.XGBoosterCreate(new IntPtr[] { dmat2 }, 1, ref handle);
            }

            //WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "objective", "reg:ranking"));

            foreach (var obj in allElements)
            {
                var features = obj.f;
                var b = DateTime.Now;
                PredictOneOff(handle, ref features, ref f3s[obj.i], ref sharedBuffer);
                c3s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
            }

            System.Threading.Tasks.Parallel.ForEach(allElements.Take(stop1),
                (obj) =>
                {
                    var features = obj.f;
                    var b = DateTime.Now;

                    IntPtr dmat = (IntPtr)0;
                    unsafe
                    {
                        WrappedXGBoostInterface.XGDMatrixCreateFromMat(features, 1, (uint)features.Length, float.NaN,
                            ref dmat);
                    }
                    PredictN(dmat, handle, ref f1s[obj.i], ref lockobj);
                    unsafe
                    {
                        WrappedXGBoostInterface.XGDMatrixFree(dmat);
                    }

                    c1s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                });

            int nbThread = 0;
            System.Threading.Tasks.Parallel.ForEach(allElements, () =>
            {
                nbThread += 1;
                return WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();
            },
                (obj, loop, buffer) =>
                {
                    // Custom API
                    var features = obj.f;
                    var b = DateTime.Now;
                    PredictOneOff(handle, ref features, ref f2s[obj.i], ref buffer);
                    c2s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                    return buffer;
                },
                _ => { });

            var issues = new List<int>();
            for (int i = 0; i < stop1; ++i)
            {
                if (f1s[i].Length != f2s[i].Length)
                {
                    issues.Add(i);
                    break;
                }
                var d = Math.Abs(f1s[i][0] - f2s[i][0]);
                if (d > 1e-5)
                {
                    issues.Add(i);
                }
            }

            unsafe
            {
                WrappedXGBoostInterface.XGBoosterFree(handle);
            }

            var s1 = c1s.Sum() / stop1;
            var s2 = c2s.Sum() / allElements.Count;
            var s3 = c3s.Sum() / allElements.Count;

            if (issues.Count > 0)
                throw new Exception(string.Join(", ", issues.Select(i => i.ToString()).ToArray()));

            Console.WriteLine("nbTh={4} s1={0} s2={1} s3={3}, N={2}", s1, s2, f1s.Length, s3, nbThread);
        }

        public static void TrainingTest()
        {
            var data = GetDataDir("housing.txt");
            var datamat = ReadCsv(data).ToList();
            IntPtr dmat = CreateDMatrix(datamat);
            IntPtr handle = (IntPtr)0;

            unsafe
            {
                WrappedXGBoostInterface.XGBoosterCreate(new IntPtr[] { dmat }, 1, ref handle);
            }

            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "silent", "1"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "nthread", "1"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "seed", "42"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "learning_rate", "0.3"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "gamma", "0"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "max_depth", "6"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "min_child_weight", "1"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "max_delta_step", "0"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "subsample", "1"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "colsample_bytree", "1"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "colsample_bylevel", "1"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "alpha", "0"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "tree_method", "auto"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "sketch_eps", "0.03"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "scale_pos_weight", "1"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "lambda", "1"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "booster", "gbtree"));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "objective", "rank:pairwise"));

            int it = 0;
            IntPtr outResult;
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(handle, it, dmat));
            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterEvalOneIter(handle, it++, new IntPtr[] { dmat }, new string[] { "ok" }, 1, out outResult));
        }

        static public void XGBoostNoThreadCompareEvaluationMCWholeFileTest()
        {
            var lockobj = new object();
            var data = GetDataDir("housing.txt");
            var model = GetDataDir("xgbcl-iris-0.6-2016-09-02.xgb");
            var bin = File.ReadAllBytes(model);
            IntPtr handle = (IntPtr)0;
            WrappedXGBoostInterface.XGBoosterCreate(null, 0, ref handle);

            var rand = new Random();
            var datamat = new float[1][];
            for (int l = 0; l < datamat.Length; ++l)
            {
                datamat[l] = new float[] { 0f, 0f, 3f, 1f, 1f, 1f };
            }
            var allElements_ = datamat.Select((c, i) => new { i = i, f = c.Skip(1).ToArray() }).ToList();
            var ac = allElements_.ToArray();
            for (int i = 0; i < 10; ++i)
                allElements_.AddRange(ac);
            var allElements = allElements_.Select((c, i) => new { i = i, f = c.f }).ToList();
            var groups = allElements.Select(c => 0f).ToArray();

            var c1s = new double[allElements.Count];
            var c3s = new double[allElements.Count];
            var f1s = new float[allElements.Count][];
            var f3s = new float[allElements.Count][];
            var sharedBuffer = WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();

            IntPtr dmat2 = CreateDMatrix(datamat, groups);

            unsafe
            {
                fixed (byte* b = bin)
                {
                    WrappedXGBoostInterface.XGBoosterLoadModelFromBuffer(handle, b, (uint)bin.Length);
                    WrappedXGBoostInterface.XGBoosterLazyInit(handle);
                }
                //WrappedXGBoostInterface.XGBoosterCreate(new IntPtr[] { dmat2 }, 1, ref handle);
            }

            //WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterSetParam(handle, "objective", "reg:ranking"));

            foreach (var margin in new[] { false, true })
            {
                var obji = new List<int>();
                foreach (var obj in allElements)
                {
                    var features = obj.f;
                    obji.Add(obj.i);

                    var b = DateTime.Now;
                    PredictOneOff(handle, ref features, ref f3s[obj.i], ref sharedBuffer, margin);
                    c3s[obj.i] = (DateTime.Now - b).TotalMilliseconds;

                    IntPtr dmat = (IntPtr)0;
                    unsafe
                    {
                        WrappedXGBoostInterface.XGDMatrixCreateFromMat(features, 1, (uint)features.Length, float.NaN,
                            ref dmat);
                    }
                    PredictN(dmat, handle, ref f1s[obj.i], ref lockobj, margin);
                    unsafe
                    {
                        WrappedXGBoostInterface.XGDMatrixFree(dmat);
                    }

                    c1s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                }
                foreach (var i in obji)
                    if (f1s[i][0] != f3s[i][0])
                        throw new Exception(string.Format("{0} != {1} - {2}", f1s[i][0], f3s[i][0], margin));
            }
        }

        public static void XGBoostNoThreadCompareEvaluationDartWholeFileTest2(int stopAfter)
        {
            var data = GetDataDir("housing.txt");
            var model = GetDataDir("xgbreg-dart-0.6-2016-08-26.xgb");
            var bin = File.ReadAllBytes(model);
            var booster = new Booster(bin, 13);
            booster.LazyInit();
            var allElements = ReadCsv(data).Select((c, i) => new { i = i, f = c.Skip(1).ToArray() }).ToList();

#if(DEBUG)
            int nb = 1;
#else
            int nb = allElements.Count;
#endif
            var c1s = new double[nb];
            var c2s = new double[nb];
            var f1s = new float[nb][];
            var f2s = new float[nb][];

            var uniqueBuffer = Booster.CreateInternalBuffer();

            // Custom API
            int loop = 0;
            foreach (var obj in allElements)
            {
                if (loop >= stopAfter)
                    break;
                ++loop;
                var features = obj.f;
                FloatVector vbuf = new FloatVector(features.Length, features);
                var b = DateTime.Now;
                FloatVector predictedValues2 = new FloatVector();
                booster.Predict(ref vbuf, ref predictedValues2, ref uniqueBuffer);
                c2s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                f2s[obj.i] = predictedValues2.Values;
                if (predictedValues2.Count != 1)
                    throw new Exception();
#if(DEBUG)
                break;
#endif
            }

            // Standard API
            loop = 0;
            foreach (var obj in allElements)
            {
                if (loop >= stopAfter)
                    break;
                ++loop;
                var features = obj.f;
                FloatVector vbuf = new FloatVector(features.Length, features);
                var b = DateTime.Now;
                var dmat = new DMatrix(features, 1, (uint)features.Length);
                FloatVector predictedValues1 = new FloatVector();
                booster.PredictN(dmat, ref predictedValues1);
                c1s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
                f1s[obj.i] = predictedValues1.Values;
#if(DEBUG)
                break;
#endif
            }

            var issues = new List<int>();
            for (int i = 0; i < f1s.Length; ++i)
            {
                if (i >= stopAfter)
                    break;
                if (f1s[i].Length != f2s[i].Length)
                {
                    issues.Add(i);
                    break;
                }
                var d = Math.Abs(f1s[i][0] - f2s[i][0]);
                if (d > 1e-5)
                    issues.Add(i);
            }

            if (issues.Count > 0)
                throw new Exception(string.Join(", ", issues));
        }

        static public void XGBooostLightModel()
        {
            var lockobj = new object();
            var model = GetDataDir("xgb_binary_11.xgb");
            var bin = File.ReadAllBytes(model);
            IntPtr handle = (IntPtr)0;
            WrappedXGBoostInterface.XGBoosterCreate(null, 0, ref handle);

            var data = GetDataDir("xgb_binary_11.csv");
            var datamat = ReadCsv(data).ToList();
            var allElements_ = datamat.Select((c, i) => new { i = i, f = c.ToArray() }).ToList();
            var allElements = allElements_;

            var c1s = new double[allElements.Count];
            var c2s = new double[allElements.Count];
            var c3s = new double[allElements.Count];
            var f1s = new float[allElements.Count][];
            var f2s = new float[allElements.Count][];
            var f3s = new float[allElements.Count][];
            var sharedBuffer = WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();

            IntPtr dmat2 = CreateDMatrix(datamat);

            unsafe
            {
                fixed (byte* b = bin)
                {
                    WrappedXGBoostInterface.XGBoosterLoadModelFromBuffer(handle, b, (uint)bin.Length);
                    WrappedXGBoostInterface.XGBoosterLazyInit(handle);
                }
            }

            WrappedXGBoostInterface.Check(WrappedXGBoostInterface.XGBoosterUpdateOneIter(handle, 0, dmat2));

            foreach (var obj in allElements)
            {
                var features = obj.f;
                IntPtr dmat = (IntPtr)0;
                unsafe
                {
                    WrappedXGBoostInterface.XGDMatrixCreateFromMat(features, 1, (uint)features.Length, float.NaN,
                        ref dmat);
                }
                PredictN(dmat, handle, ref f1s[obj.i], ref lockobj, false);

                var b = DateTime.Now;
                PredictOneOff(handle, ref features, ref f3s[obj.i], ref sharedBuffer);
                c3s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
            }

            var ntr = WrappedXGBoostInterface.XGBoosterGetNumInfo(handle, "NumTrees");
            if (ntr <= 1)
                throw new Exception("no tree");
        }

        static public void XGBooostLightModel2()
        {
            var lockobj = new object();
            var model = GetDataDir("xgb_multi_11.xgb");
            var bin = File.ReadAllBytes(model);
            IntPtr handle = (IntPtr)0;
            WrappedXGBoostInterface.XGBoosterCreate(null, 0, ref handle);

            var data = GetDataDir("xgb_binary_11.csv");
            var datamat = ReadCsv(data).ToList();
            var allElements_ = datamat.Select((c, i) => new { i = i, f = c.ToArray() }).ToList();
            var allElements = allElements_;

            var c1s = new double[allElements.Count];
            var c2s = new double[allElements.Count];
            var c3s = new double[allElements.Count];
            var f1s = new float[allElements.Count][];
            var f2s = new float[allElements.Count][];
            var f3s = new float[allElements.Count][];
            var sharedBuffer = WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();

            IntPtr dmat2 = CreateDMatrix(datamat);

            unsafe
            {
                fixed (byte* b = bin)
                {
                    WrappedXGBoostInterface.XGBoosterLoadModelFromBuffer(handle, b, (uint)bin.Length);
                    WrappedXGBoostInterface.XGBoosterLazyInit(handle);
                }
            }

            foreach (var obj in allElements)
            {
                var features = obj.f;
                IntPtr dmat = (IntPtr)0;
                unsafe
                {
                    WrappedXGBoostInterface.XGDMatrixCreateFromMat(features, 1, (uint)features.Length, float.NaN,
                        ref dmat);
                }
                PredictN(dmat, handle, ref f1s[obj.i], ref lockobj, false);

                var b = DateTime.Now;
                PredictOneOff(handle, ref features, ref f3s[obj.i], ref sharedBuffer);
                c3s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
            }
        }

        static public void XGBooostLightModel3()
        {
            var lockobj = new object();
            var model = GetDataDir("xgb_reg_iris12.xgb");
            var bin = File.ReadAllBytes(model);
            IntPtr handle = (IntPtr)0;
            WrappedXGBoostInterface.XGBoosterCreate(null, 0, ref handle);

            var data = GetDataDir("xgb_reg_iris22.csv");
            var datamat = ReadCsv(data).ToList();
            var allElements_ = datamat.Select((c, i) => new { i = i, f = c.ToArray() }).ToList();
            var allElements = allElements_;

            var c1s = new double[allElements.Count];
            var c2s = new double[allElements.Count];
            var c3s = new double[allElements.Count];
            var f1s = new float[allElements.Count][];
            var f2s = new float[allElements.Count][];
            var f3s = new float[allElements.Count][];
            var sharedBuffer = WrappedXGBoostInterface.XGBoostTreeBuffer.CreateInternalBuffer();

            IntPtr dmat2 = CreateDMatrix(datamat);

            unsafe
            {
                fixed (byte* b = bin)
                {
                    WrappedXGBoostInterface.XGBoosterLoadModelFromBuffer(handle, b, (uint)bin.Length);
                    WrappedXGBoostInterface.XGBoosterLazyInit(handle);
                }
            }

            foreach (var obj in allElements)
            {
                var features = obj.f;
                IntPtr dmat = (IntPtr)0;
                unsafe
                {
                    WrappedXGBoostInterface.XGDMatrixCreateFromMat(features, 1, (uint)features.Length, float.NaN,
                        ref dmat);
                }
                PredictN(dmat, handle, ref f1s[obj.i], ref lockobj, false);

                var b = DateTime.Now;
                PredictOneOff(handle, ref features, ref f3s[obj.i], ref sharedBuffer);
                c3s[obj.i] = (DateTime.Now - b).TotalMilliseconds;
            }
        }
    }
}
