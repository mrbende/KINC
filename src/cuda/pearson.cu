





extern "C" __device__ float Pearson_computeCluster(
   const float2 *data,
   const char *labels,
   int sampleSize,
   char cluster,
   int minSamples)
{
   // compute intermediate sums
   int n = 0;
   float sumx = 0;
   float sumy = 0;
   float sumx2 = 0;
   float sumy2 = 0;
   float sumxy = 0;

   for ( int i = 0, j = 0; i < sampleSize; ++i )
   {
      if ( labels[i] >= 0 )
      {
         if ( labels[i] == cluster )
         {
            float x_i = data[j].x;
            float y_i = data[j].y;

            sumx += x_i;
            sumy += y_i;
            sumx2 += x_i * x_i;
            sumy2 += y_i * y_i;
            sumxy += x_i * y_i;

            ++n;
         }

         ++j;
      }
   }

   // compute correlation only if there are enough samples
   float result = NAN;

   if ( n >= minSamples )
   {
      result = (n*sumxy - sumx*sumy) / sqrt((n*sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy));
   }

   return result;
}






extern "C" __global__ void Pearson_compute(
   const float2 *in_data,
   char clusterSize,
   const char *in_labels,
   int sampleSize,
   int minSamples,
   float *out_correlations)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   const float2 *data = &in_data[i * sampleSize];
   const char *labels = &in_labels[i * sampleSize];
   float *correlations = &out_correlations[i * clusterSize];

   for ( char k = 0; k < clusterSize; ++k )
   {
      correlations[k] = Pearson_computeCluster(data, labels, sampleSize, k, minSamples);
   }
}
