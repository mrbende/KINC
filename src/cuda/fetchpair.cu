
// #include "linalg.cu"






extern "C" __global__ void fetchPair(
   const float *expressions,
   int sampleSize,
   const int2 *in_index,
   int minExpression,
   Vector2 *out_X,
   int *out_N,
   char *out_labels)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   // initialize variables
   int2 index = in_index[i];
   Vector2 *X = &out_X[i * sampleSize];
   char *labels = &out_labels[i * sampleSize];
   int *p_numSamples = &out_N[i];

   if ( index.x == 0 && index.y == 0 )
   {
      return;
   }

   // index into gene expressions
   const float *gene1 = &expressions[index.x * sampleSize];
   const float *gene2 = &expressions[index.y * sampleSize];

   // populate X with shared expressions of gene pair
   int numSamples = 0;

   for ( int i = 0; i < sampleSize; ++i )
   {
      if ( isnan(gene1[i]) || isnan(gene2[i]) )
      {
         labels[i] = -9;
      }
      else if ( gene1[i] < minExpression || gene2[i] < minExpression )
      {
         labels[i] = -6;
      }
      else
      {
         X[numSamples] = make_float2(gene1[i], gene2[i]);
         numSamples++;

         labels[i] = 0;
      }
   }

   // return size of X
   *p_numSamples = numSamples;
}
