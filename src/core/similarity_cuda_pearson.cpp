#include "similarity_cuda_pearson.h"



using namespace std;






Similarity::CUDA::Pearson::Pearson(::CUDA::Program* program):
   ::CUDA::Kernel(program, "Pearson_compute")
{
}






::CUDA::Event Similarity::CUDA::Pearson::execute(
   const ::CUDA::Stream& stream,
   int kernelSize,
   ::CUDA::Buffer<float2>* in_data,
   char clusterSize,
   ::CUDA::Buffer<qint8>* in_labels,
   int sampleSize,
   int minSamples,
   ::CUDA::Buffer<float>* out_correlations
)
{
   // set kernel arguments
   setBuffer(InData, in_data);
   setArgument(ClusterSize, clusterSize);
   setBuffer(InLabels, in_labels);
   setArgument(SampleSize, sampleSize);
   setArgument(MinSamples, minSamples);
   setBuffer(OutCorrelations, out_correlations);

   // set kernel sizes
   int blockSize {getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)};

   setGridSize((kernelSize + blockSize - 1) / blockSize);
   setBlockSize(blockSize);

   // execute kernel
   return ::CUDA::Kernel::execute(stream);
}
