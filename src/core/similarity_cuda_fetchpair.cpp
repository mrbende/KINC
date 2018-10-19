#include "similarity_cuda_fetchpair.h"



using namespace std;






Similarity::CUDA::FetchPair::FetchPair(::CUDA::Program* program):
   ::CUDA::Kernel(program, "fetchPair")
{
}






::CUDA::Event Similarity::CUDA::FetchPair::execute(
   const ::CUDA::Stream& stream,
   int kernelSize,
   ::CUDA::Buffer<float>* expressions,
   int sampleSize,
   ::CUDA::Buffer<int2>* in_index,
   int minExpression,
   ::CUDA::Buffer<float2>* out_X,
   ::CUDA::Buffer<int>* out_N,
   ::CUDA::Buffer<qint8>* out_labels
)
{
   // set kernel arguments
   setBuffer(Expressions, expressions);
   setArgument(SampleSize, sampleSize);
   setBuffer(InIndex, in_index);
   setArgument(MinExpression, minExpression);
   setBuffer(OutX, out_X);
   setBuffer(OutN, out_N);
   setBuffer(OutLabels, out_labels);

   // set kernel sizes
   int blockSize {getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)};

   setGridSize((kernelSize + blockSize - 1) / blockSize);
   setBlockSize(blockSize);

   // execute kernel
   return ::CUDA::Kernel::execute(stream);
}
