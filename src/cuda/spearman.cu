
// #include "sort.cu"






extern "C" __device__ int nextPower2(int n)
{
	int pow2 = 2;
	while ( pow2 < n )
	{
		pow2 *= 2;
	}

	return pow2;
}






extern "C" __device__ float Spearman_computeCluster(
   const float2 *data,
   const char *labels,
	int sampleSize,
   char cluster,
   int minSamples,
   float *x,
   float *y,
   int *rank)
{
   // extract samples in gene pair cluster
   int N_pow2 = nextPower2(sampleSize);
	int n = 0;

	for ( int i = 0, j = 0; i < sampleSize; ++i )
   {
      if ( labels[i] >= 0 )
      {
         if ( labels[i] == cluster )
         {
            x[n] = data[j].x;
            y[n] = data[j].y;
				rank[n] = n + 1;
            ++n;
         }

         ++j;
      }
   }

   for ( int i = n; i < N_pow2; ++i )
   {
      x[i] = INFINITY;
      y[i] = INFINITY;
      rank[i] = 0;
   }

   // compute correlation only if there are enough samples
   float result = NAN;

   if ( n >= minSamples )
   {
      // get new power of 2 floor size
      int n_pow2 = nextPower2(n);

      // execute two bitonic sorts that is beginning of spearman algorithm
      bitonicSortFF(n_pow2, x, y);
      bitonicSortFI(n_pow2, y, rank);

      // go through spearman sorted rank list and calculate difference from 1,2,3,... list
      int diff = 0;

      for ( int i = 0; i < n; ++i )
      {
         int tmp = (i + 1) - rank[i];
         diff += tmp*tmp;
      }

      // compute spearman coefficient
      result = 1.0 - 6.0 * diff / (n * (n*n - 1));
   }

   return result;
}






extern "C" __global__ void Spearman_compute(
   const float2 *in_data,
   char clusterSize,
   const char *in_labels,
	int sampleSize,
   int minSamples,
   float *work_x,
   float *work_y,
   int *work_rank,
   float *out_correlations)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
	int N_pow2 = nextPower2(sampleSize);

   const float2 *data = &in_data[i * sampleSize];
   const char *labels = &in_labels[i * sampleSize];
   float *x = &work_x[i * N_pow2];
   float *y = &work_y[i * N_pow2];
   int *rank = &work_rank[i * N_pow2];
   float *correlations = &out_correlations[i * clusterSize];

   for ( char k = 0; k < clusterSize; ++k )
   {
      correlations[k] = Spearman_computeCluster(data, labels, sampleSize, k, minSamples, x, y, rank);
   }
}
