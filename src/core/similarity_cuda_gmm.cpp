#include "similarity_cuda_gmm.h"



using namespace std;






Similarity::CUDA::GMM::GMM(::CUDA::Program* program):
   ::CUDA::Kernel(program, "GMM_compute")
{
}






::CUDA::Event Similarity::CUDA::GMM::execute(
   const ::CUDA::Stream& stream,
   int kernelSize,
   int sampleSize,
   int minSamples,
   char minClusters,
   char maxClusters,
   int criterion,
   int removePreOutliers,
   int removePostOutliers,
   ::CUDA::Buffer<float2>* work_X,
   ::CUDA::Buffer<int>* work_N,
   ::CUDA::Buffer<float>* work_x,
   ::CUDA::Buffer<float>* work_y,
   ::CUDA::Buffer<qint8>* work_labels,
   ::CUDA::Buffer<cu_component>* work_components,
   ::CUDA::Buffer<float2>* work_MP,
   ::CUDA::Buffer<int>* work_counts,
   ::CUDA::Buffer<float>* work_logpi,
   ::CUDA::Buffer<float>* work_loggamma,
   ::CUDA::Buffer<float>* work_logGamma,
   ::CUDA::Buffer<qint8>* out_K,
   ::CUDA::Buffer<qint8>* out_labels
)
{
   // set kernel arguments
   setArgument(SampleSize, sampleSize);
   setArgument(MinSamples, minSamples);
   setArgument(MinClusters, minClusters);
   setArgument(MaxClusters, maxClusters);
   setArgument(Criterion, criterion);
   setArgument(RemovePreOutliers, removePreOutliers);
   setArgument(RemovePostOutliers, removePostOutliers);
   setBuffer(WorkX, work_X);
   setBuffer(WorkN, work_N);
   setBuffer(WorkXSorted, work_x);
   setBuffer(WorkYSorted, work_y);
   setBuffer(WorkLabels, work_labels);
   setBuffer(WorkComponents, work_components);
   setBuffer(WorkMP, work_MP);
   setBuffer(WorkCounts, work_counts);
   setBuffer(WorkLogPi, work_logpi);
   setBuffer(WorkLoggamma, work_loggamma);
   setBuffer(WorkLogGamma, work_logGamma);
   setBuffer(OutK, out_K);
   setBuffer(OutLabels, out_labels);

   // set kernel sizes
   int blockSize {getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)};

   setGridSize((kernelSize + blockSize - 1) / blockSize);
   setBlockSize(blockSize);

   // execute kernel
   return ::CUDA::Kernel::execute(stream);
}
