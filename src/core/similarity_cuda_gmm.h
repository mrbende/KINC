#ifndef SIMILARITY_CUDA_GMM_H
#define SIMILARITY_CUDA_GMM_H
#include "similarity_cuda.h"



typedef struct
{
   float pi;
   float2 mu;
   float4 sigma;
   float4 sigmaInv;
   float normalizer;
} cu_component;






class Similarity::CUDA::GMM : public ::CUDA::Kernel
{
public:
   enum Argument
   {
      SampleSize
      ,MinSamples
      ,MinClusters
      ,MaxClusters
      ,Criterion
      ,RemovePreOutliers
      ,RemovePostOutliers
      ,WorkX
      ,WorkN
      ,WorkXSorted
      ,WorkYSorted
      ,WorkLabels
      ,WorkComponents
      ,WorkMP
      ,WorkCounts
      ,WorkLogPi
      ,WorkLoggamma
      ,WorkLogGamma
      ,OutK
      ,OutLabels
   };
   explicit GMM(::CUDA::Program* program);
   ::CUDA::Event execute(
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
   );
};



#endif
