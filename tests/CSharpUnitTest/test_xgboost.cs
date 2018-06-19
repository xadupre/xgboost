using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;


namespace CSharpUnitTest
{
    [TestClass]
    public class test_xgboost
    {
        public static void CheckDllPath()
        {
            var name = WrappedXGBoostInterface.DllName;
            if (!File.Exists(name))
            {
                var full = Path.GetFullPath(name);
                throw new FileNotFoundException(full);
            }
        }

        [TestMethod]
        public void XGBoostNoThreadCompareEvaluationMCWholeFileTest()
        {
            CheckDllPath();
            xgb_test.XGBoostNoThreadCompareEvaluationMCWholeFileTest();
        }

        [TestMethod]
        public void XGBoostCompareEvaluationBinaryWholeFileTest()
        {
            CheckDllPath();
            xgb_test.XGBoostCompareEvaluationBinaryWholeFileTest();
        }

        [TestMethod]
        public void XGBoostCompareEvaluationRankingWholeFileTest()
        {
            CheckDllPath();
            xgb_test.XGBoostCompareEvaluationRankingWholeFileTest();
        }

        [TestMethod]
        public void XGBoostCompareEvaluationDartWholeFileTest()
        {
            CheckDllPath();
            xgb_test.XGBoostCompareEvaluationDartWholeFileTest();
        }

        [TestMethod]
        public void XGBoostNoThreadCompareEvaluationDartWholeFileTest2()
        {
            CheckDllPath();
            xgb_test.XGBoostNoThreadCompareEvaluationDartWholeFileTest2(1);
            xgb_test.XGBoostNoThreadCompareEvaluationDartWholeFileTest2(1000);
        }

        [TestMethod]
        public void XGBoostTrainingTest()
        {
            CheckDllPath();
            xgb_test.TrainingTest();
        }

        [TestMethod]
        public void XGBooostLightModel()
        {
            CheckDllPath();
            xgb_test.XGBooostLightModel();
        }

        [TestMethod]
        public void XGBooostLightModel2()
        {
            CheckDllPath();
            xgb_test.XGBooostLightModel2();
        }

        [TestMethod]
        public void XGBooostLightModel3()
        {
            CheckDllPath();
            xgb_test.XGBooostLightModel3();
        }
    }
}
