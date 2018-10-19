#ifndef SIMILARITY_CUDA_PEARSON_H
#define SIMILARITY_CUDA_PEARSON_H
#include "similarity_cuda.h"



class Similarity::CUDA::Pearson : public ::CUDA::Kernel
{
public:
   enum Argument
   {
      InData
      ,ClusterSize
      ,InLabels
      ,SampleSize
      ,MinSamples
      ,OutCorrelations
   };
   explicit Pearson(::CUDA::Program* program);
   ::CUDA::Event execute(
      const ::CUDA::Stream& stream,
      int kernelSize,
      ::CUDA::Buffer<float2>* in_data,
      char clusterSize,
      ::CUDA::Buffer<qint8>* in_labels,
      int sampleSize,
      int minSamples,
      ::CUDA::Buffer<float>* out_correlations
   );
};



#endif
