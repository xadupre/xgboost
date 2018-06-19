using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSharpUnitTest
{
    public class FloatVector
    {
        public float[] Values;
        public int[] Indices;
        public int Length { get; }
        public int Count { get; }

        public bool IsDense { get { return Indices == null; } }

        public FloatVector()
        {
            Values = null;
            Indices = null;
            Length = 0;
            Count = 0;
        }

        public FloatVector(float[] values)
        {
            Values = values;
            Indices = null;
            Length = Values.Length;
            Count = Values.Length;
        }

        public FloatVector(int dim, float[] values)
        {
            Values = values;
            Indices = null;
            Count = dim;
            Length = Values.Length;
        }
    }
}
