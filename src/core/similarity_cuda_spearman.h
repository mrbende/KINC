#ifndef SIMILARITY_CUDA_SPEARMAN_H
#define SIMILARITY_CUDA_SPEARMAN_H
#include "similarity_cuda.h"



class Similarity::CUDA::Spearman : public ::CUDA::Kernel
{
public:
   enum Argument
   {
      InData
      ,ClusterSize
      ,InLabels
      ,SampleSize
      ,MinSamples
      ,WorkX
      ,WorkY
      ,WorkRank
      ,OutCorrelations
   };
   explicit Spearman(::CUDA::Program* program);
   ::CUDA::Event execute(
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
   );
};



#endif
