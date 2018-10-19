#include "similarity_cuda_spearman.h"



using namespace std;






Similarity::CUDA::Spearman::Spearman(::CUDA::Program* program):
   ::CUDA::Kernel(program, "Spearman_compute")
{
}






::CUDA::Event Similarity::CUDA::Spearman::execute(
   const ::CUDA::Stream& stream,
   int kernelSize,
   ::CUDA::Buffer<float2>* in_data,
   char clusterSize,
   ::CUDA::Buffer<qint8>* in_labels,
   int sampleSize,
   int minSamples,
   ::CUDA::Buffer<float>* work_x,
   ::CUDA::Buffer<float>* work_y,
   ::CUDA::Buffer<int>* work_rank,
   ::CUDA::Buffer<float>* out_correlations
)
{
   // set kernel arguments
   setBuffer(InData, in_data);
   setArgument(ClusterSize, clusterSize);
   setBuffer(InLabels, in_labels);
   setArgument(SampleSize, sampleSize);
   setArgument(MinSamples, minSamples);
   setBuffer(WorkX, work_x);
   setBuffer(WorkY, work_y);
   setBuffer(WorkRank, work_rank);
   setBuffer(OutCorrelations, out_correlations);

   // set kernel sizes
   int blockSize {getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)};

   setGridSize((kernelSize + blockSize - 1) / blockSize);
   setBlockSize(blockSize);

   // execute kernel
   return ::CUDA::Kernel::execute(stream);
}
