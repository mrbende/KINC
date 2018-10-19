
// #include "sort.cu"






/*!
 * Implementation of rand(), taken from POSIX example.
 *
 * @param state
 */
extern "C" __device__ int rand(unsigned long *state)
{
   *state = (*state) * 1103515245 + 12345;
   return ((unsigned)((*state)/65536) % 32768);
}





extern "C" __device__ int removeOutliers(
   Vector2 *data,
   char *labels,
   int sampleSize,
   char cluster,
   char marker,
   float *x_sorted,
   float *y_sorted)
{
   // extract univariate data from the given cluster
   int n = 0;

   for ( int i = 0, j = 0; i < sampleSize; i++ )
   {
      if ( labels[i] >= 0 )
      {
         if ( labels[i] == cluster )
         {
            x_sorted[n] = data[j].x;
            y_sorted[n] = data[j].y;
            n++;
         }

         j++;
      }
   }

   // return if the given cluster is empty
   if ( n == 0 )
   {
      return 0;
   }

   // sort samples for each axis
   heapSort(x_sorted, n);
   heapSort(y_sorted, n);

   // compute interquartile range and thresholds for each axis
   float Q1_x = x_sorted[n * 1 / 4];
   float Q3_x = x_sorted[n * 3 / 4];
   float T_x_min = Q1_x - 1.5f * (Q3_x - Q1_x);
   float T_x_max = Q3_x + 1.5f * (Q3_x - Q1_x);

   float Q1_y = y_sorted[n * 1 / 4];
   float Q3_y = y_sorted[n * 3 / 4];
   float T_y_min = Q1_y - 1.5f * (Q3_y - Q1_y);
   float T_y_max = Q3_y + 1.5f * (Q3_y - Q1_y);

   // remove outliers
   int numSamples = 0;

   for ( int i = 0, j = 0; i < sampleSize; i++ )
   {
      if ( labels[i] >= 0 )
      {
         // mark samples in the given cluster that are outliers on either axis
         if ( labels[i] == cluster && (data[j].x < T_x_min || T_x_max < data[j].x || data[j].y < T_y_min || T_y_max < data[j].y) )
         {
            labels[i] = marker;
         }

         // preserve all other non-outlier samples in the data array
         else
         {
            data[numSamples] = data[j];
            numSamples++;
         }

         j++;
      }
   }

   // return number of remaining samples
   return numSamples;
}
