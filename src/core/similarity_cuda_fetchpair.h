#ifndef SIMILARITY_CUDA_FETCHPAIR_H
#define SIMILARITY_CUDA_FETCHPAIR_H
#include "similarity_cuda.h"



class Similarity::CUDA::FetchPair : public ::CUDA::Kernel
{
public:
   enum Argument
   {
      Expressions
      ,SampleSize
      ,InIndex
      ,MinExpression
      ,OutX
      ,OutN
      ,OutLabels
   };
   explicit FetchPair(::CUDA::Program* program);
   ::CUDA::Event execute(
      const ::CUDA::Stream& stream,
      int kernelSize,
      ::CUDA::Buffer<float>* expressions,
      int sampleSize,
      ::CUDA::Buffer<int2>* in_index,
      int minExpression,
      ::CUDA::Buffer<float2>* out_X,
      ::CUDA::Buffer<int>* out_N,
      ::CUDA::Buffer<qint8>* out_labels
   );
};



#endif
