
void vectorInitZero(__global float2 *a)
{
   a->x = 0;
   a->y = 0;
}






void vectorAdd(__global float2 *a, const __global float2 *b)
{
   a->x += b->x;
   a->y += b->y;
}






void vectorScale(__global float2 *a, float c)
{
   a->x *= c;
   a->y *= c;
}






float vectorDiffNorm(const __global float2 *a, const __global float2 *b)
{
   float dist = 0;
   dist += (a->x - b->x) * (a->x - b->x);
   dist += (a->y - b->y) * (a->y - b->y);

   return sqrt(dist);
}






/**
 * Implementation of the MWC64X random number generator.
 *
 * @param state
 */
uint rand(uint2 *state)
{
   enum { A = 4294883355U };
   uint x = (*state).x, c = (*state).y;  // Unpack the state
   uint res = x ^ c;                     // Calculate the result
   uint hi = mul_hi(x, A);               // Step the RNG
   x = x * A + c;
   c = hi + (x < c);
   *state = (uint2)(x, c);               // Pack the state back up

   return res;                           // Return the next result
}





/**
 * Fetch and build data matrix X for a pair of genes, skipping any expressions
 * that are missing for either gene.
 *
 * @param expressions
 * @param size
 * @param indexA
 * @param indexB
 * @param X
 * @return number of rows in X
 */
int fetchData(
   __global const float *expressions, int size,
   int indexA, int indexB,
   __global float2 *X)
{
   int numSamples = 0;

   indexA *= size;
   indexB *= size;

   // build data matrix from expressions and indices
   int i;
   for ( i = 0; i < size; ++i )
   {
      if ( !isnan(expressions[indexA + i]) && !isnan(expressions[indexB + i]) )
      {
         // if both expressions exist add expressions to new lists and increment
         X[numSamples] = (float2) (
            expressions[indexA + i],
            expressions[indexB + i]
         );
         numSamples++;
      }
   }

   // return size of X
   return numSamples;
}






/**
 * Compute the log-likelihood of a K-means model given data X.
 *
 * @param X
 * @param N
 * @param y
 * @param Mu
 * @param K
 */
float computeLogLikelihood(
   __global const float2 *X, int N,
   __global const int *y,
   __global const float2 *Mu, int K)
{
   // compute within-class scatter
   float S = 0;

   for ( int k = 0; k < K; ++k )
   {
      for ( int i = 0; i < N; ++i )
      {
         if ( y[i] != k )
         {
            continue;
         }

         float dist = vectorDiffNorm(&X[i], &Mu[k]);

         S += dist * dist;
      }
   }

   return -S;
}






/**
 * Compute a K-means clustering model from a dataset.
 *
 * @param X
 * @param N
 * @param y
 * @param y_next
 * @param Mu
 * @param K
 * @param logL
 */
void computeKmeans(
   __global const float2 *X, int N,
   __global int *y,
   __global int *y_next,
   __global float2 *Mu, int K,
   float *logL)
{
   uint2 state = (get_global_id(0), get_global_id(1));

   // initialize means randomly from X
   int k;
   for ( k = 0; k < K; ++k )
   {
      int i = rand(&state) % N;
      Mu[k] = X[i];
   }

   // iterate K means until convergence
   while ( true )
   {
      // compute new labels
      int i;
      for ( i = 0; i < N; ++i )
      {
         // find k that minimizes norm(x_i - mu_k)
         int min_k = -1;
         float min_dist;

         for ( k = 0; k < K; ++k )
         {
            float dist = vectorDiffNorm(&X[i], &Mu[k]);

            if ( min_k == -1 || dist < min_dist )
            {
               min_k = k;
               min_dist = dist;
            }
         }

         y_next[i] = min_k;
      }

      // check for convergence
      bool converged = true;

      for ( i = 0; i < N; ++i )
      {
         if ( y[i] != y_next[i] )
         {
            converged = false;
            break;
         }
      }

      if ( converged )
      {
         break;
      }

      // update labels
      for ( i = 0; i < N; ++i )
      {
         y[i] = y_next[i];
      }

      // update means
      for ( k = 0; k < K; ++k )
      {
         // compute mu_k = mean of all x_i in cluster k
         int n_k = 0;

         vectorInitZero(&Mu[k]);

         for ( i = 0; i < N; ++i )
         {
            if ( y[i] == k )
            {
               vectorAdd(&Mu[k], &X[i]);
               n_k++;
            }
         }

         vectorScale(&Mu[k], 1.0f / n_k);
      }
   }

   *logL = computeLogLikelihood(X, N, y, Mu, K);
}






/**
 * Compute the Bayes information criterion of a K-means model.
 *
 * @param K
 * @param logL
 * @param N
 * @param D
 */
float computeBIC(int K, float logL, int N, int D)
{
   int p = K * D;

   return log((float) N) * p - 2 * logL;
}






/**
 * Compute a block of K-means models given a block of gene pairs.
 *
 * For each gene pair, several models are computed and the best model
 * is selected according to a criterion (BIC). The resulting sample mask
 * for each pair is returned.
 *
 * @param expressions
 * @param size
 * @param pairs
 * @param minSamples
 * @param minClusters
 * @param maxClusters
 * @param workX
 * @param workMu
 * @param worky1
 * @param worky2
 * @param results
 */
__kernel void computeKmeansBlock(
   __global const float *expressions, int size,
   __global const int2 *pairs,
   int minSamples, int minClusters, int maxClusters,
   __global float2 *workX,
   __global float2 *workMu,
   __global int *worky1,
   __global int *worky2,
   __global int *resultKs,
   __global int *resultLabels)
{
   // initialize workspace variables
   int i = get_global_id(0);
   __global float2 *X = &workX[i * size];
   __global float2 *Mu = &workMu[i * maxClusters];
   __global int *y1 = &worky1[i * size];
   __global int *y2 = &worky2[i * size];
   __global int *bestK = &resultKs[i];
   __global int *bestLabels = &resultLabels[i * size];

   // fetch data matrix X from expression matrix
   int numSamples = fetchData(expressions, size, pairs[i].x, pairs[i].y, X);

   // make sure minimum number of related samples is reached
   if ( numSamples >= minSamples )
   {
      float bestValue = INFINITY;

      int K;
      for ( K = minClusters; K <= maxClusters; K++ )
      {
         // run each clustering model
         float logL;
         computeKmeans(X, numSamples, y1, y2, Mu, K, &logL);

         // evaluate model
         float value = computeBIC(K, logL, numSamples, 2);

         // save the best model
         if ( value < bestValue )
         {
            *bestK = K;
            bestValue = value;

            int i;
            for ( i = 0; i < numSamples; i++ )
            {
               bestLabels[i] = y1[i];
            }
         }
      }
   }
   else
   {
      bestLabels[0] = NAN;
   }
}