#include "similarity_cuda.h"
#include <QVector>
#include "similarity_cuda_worker.h"



using namespace std;






Similarity::CUDA::CUDA(Similarity* parent):
   EAbstractAnalytic::CUDA(parent),
   _base(parent)
{
}






std::unique_ptr<EAbstractAnalytic::CUDA::Worker> Similarity::CUDA::makeWorker()
{
   return unique_ptr<EAbstractAnalytic::CUDA::Worker>(new Worker(_base, this, _program));
}






void Similarity::CUDA::initialize(::CUDA::Context* context)
{
   // create list of cuda source files
   QStringList paths {
      ":/cuda/linalg.cu",
      ":/cuda/fetchpair.cu",
      ":/cuda/sort.cu",
      ":/cuda/outlier.cu",
      ":/cuda/gmm.cu",
      ":/cuda/pearson.cu",
      ":/cuda/spearman.cu"
   };

   // create program
   _context = context;
   _program = new ::CUDA::Program(paths, this);

   // create buffer for expression data
   QVector<float> rawData = _base->_input->dumpRawData();
   _expressions = ::CUDA::Buffer<float>(rawData.size());

   // copy expression data to device
   for ( int i = 0; i < rawData.size(); ++i )
   {
      _expressions[i] = rawData[i];
   }

   _expressions.write().wait();
}
