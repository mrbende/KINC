#ifndef SIMILARITY_OPENCL_GMM_H
#define SIMILARITY_OPENCL_GMM_H
#include "similarity_opencl.h"



class Similarity::OpenCL::GMM : public ::OpenCL::Kernel
{
   Q_OBJECT
public:
   enum Argument
   {
      Expressions
      ,SampleSize
      ,MinSamples
      ,MinClusters
      ,MaxClusters
      ,Criterion
      ,RemovePreOutliers
      ,RemovePostOutliers
      ,WorkX
      ,WorkN
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
   explicit GMM(::OpenCL::Program* program, QObject* parent = nullptr);
   ::OpenCL::Event execute(
      ::OpenCL::CommandQueue* queue,
      int kernelSize,
      ::OpenCL::Buffer<cl_float>* expressions,
      cl_int sampleSize,
      cl_int minSamples,
      cl_char minClusters,
      cl_char maxClusters,
      Pairwise::Criterion criterion,
      cl_int removePreOutliers,
      cl_int removePostOutliers,
      ::OpenCL::Buffer<Pairwise::Vector2>* work_X,
      ::OpenCL::Buffer<cl_int>* work_N,
      ::OpenCL::Buffer<cl_char>* work_labels,
      ::OpenCL::Buffer<Pairwise::GMM::Component>* work_components,
      ::OpenCL::Buffer<Pairwise::Vector2>* work_MP,
      ::OpenCL::Buffer<cl_int>* work_counts,
      ::OpenCL::Buffer<cl_float>* work_logpi,
      ::OpenCL::Buffer<cl_float>* work_loggamma,
      ::OpenCL::Buffer<cl_float>* work_logGamma,
      ::OpenCL::Buffer<cl_char>* out_K,
      ::OpenCL::Buffer<cl_char>* out_labels
   );
};



#endif
