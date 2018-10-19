#ifndef SIMILARITY_CUDA_WORKER_H
#define SIMILARITY_CUDA_WORKER_H
#include "similarity_cuda.h"
#include "similarity_cuda_fetchpair.h"
#include "similarity_cuda_gmm.h"
#include "similarity_cuda_pearson.h"
#include "similarity_cuda_spearman.h"



class Similarity::CUDA::Worker : public EAbstractAnalytic::CUDA::Worker
{
   Q_OBJECT
public:
   explicit Worker(Similarity* base, Similarity::CUDA* baseCuda, ::CUDA::Program* program);
   virtual std::unique_ptr<EAbstractAnalytic::Block> execute(const EAbstractAnalytic::Block* block) override final;
private:
   Similarity* _base;
   Similarity::CUDA* _baseCuda;
   ::CUDA::Stream _stream;

   struct
   {
      CUDA::FetchPair fetchPair;
      CUDA::GMM gmm;
      CUDA::Pearson pearson;
      CUDA::Spearman spearman;
   } _kernels;

   struct
   {
      ::CUDA::Buffer<int2> in_index;
      ::CUDA::Buffer<float2> work_X;
      ::CUDA::Buffer<int> work_N;
      ::CUDA::Buffer<float> work_x;
      ::CUDA::Buffer<float> work_y;
      ::CUDA::Buffer<qint8> work_labels;
      ::CUDA::Buffer<cu_component> work_components;
      ::CUDA::Buffer<float2> work_MP;
      ::CUDA::Buffer<int> work_counts;
      ::CUDA::Buffer<float> work_logpi;
      ::CUDA::Buffer<float> work_loggamma;
      ::CUDA::Buffer<float> work_logGamma;
      ::CUDA::Buffer<int> work_rank;
      ::CUDA::Buffer<qint8> out_K;
      ::CUDA::Buffer<qint8> out_labels;
      ::CUDA::Buffer<float> out_correlations;
   } _buffers;
};



#endif
