#ifndef SIMILARITY_CUDA_H
#define SIMILARITY_CUDA_H
#include <ace/core/cudaxx.h>

#include "similarity.h"



class Similarity::CUDA : public EAbstractAnalytic::CUDA
{
   Q_OBJECT
public:
   class FetchPair;
   class GMM;
   class Pearson;
   class Spearman;
   class Worker;
   explicit CUDA(Similarity* parent);
   virtual std::unique_ptr<EAbstractAnalytic::CUDA::Worker> makeWorker() override final;
   virtual void initialize(::CUDA::Context* context) override final;
private:
   Similarity* _base;
   ::CUDA::Context* _context {nullptr};
   ::CUDA::Program* _program {nullptr};

   ::CUDA::Buffer<float> _expressions;
};



#endif
