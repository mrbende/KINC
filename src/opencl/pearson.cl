





float Pearson_computeCluster(
   __global const float2 *data,
   __global const char *labels, int N,
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

   for ( int i = 0; i < N; ++i )
   {
      if ( labels[i] == cluster )
      {
         float x_i = data[i].x;
         float y_i = data[i].y;

         sumx += x_i;
         sumy += y_i;
         sumx2 += x_i * x_i;
         sumy2 += y_i * y_i;
         sumxy += x_i * y_i;

         ++n;
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






__kernel void Pearson_compute(
   __global const float2 *in_data,
   char clusterSize,
   __global const char *in_labels,
   int sampleSize,
   int minSamples,
   __global float *out_correlations)
{
   int i = get_global_id(0);

   __global const float2 *data = &in_data[i * sampleSize];
   __global const char *labels = &in_labels[i * sampleSize];
   __global float *correlations = &out_correlations[i * clusterSize];

   for ( char k = 0; k < clusterSize; ++k )
   {
      correlations[k] = Pearson_computeCluster(data, labels, sampleSize, k, minSamples);
   }
}
