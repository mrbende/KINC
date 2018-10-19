#include "similarity_cuda_worker.h"
#include "similarity_resultblock.h"
#include "similarity_workblock.h"
#include <ace/core/elog.h>
#include "pairwise_spearman.h"



using namespace std;







Similarity::CUDA::Worker::Worker(Similarity* base, Similarity::CUDA* baseCuda, ::CUDA::Program* program):
   _base(base),
   _baseCuda(baseCuda),
   _kernels({
      .fetchPair = CUDA::FetchPair(program),
      .gmm = CUDA::GMM(program),
      .pearson = CUDA::Pearson(program),
      .spearman = CUDA::Spearman(program)
   })
{
   // initialize buffers
   int kernelSize {_base->_kernelSize};
   int N {_base->_input->sampleSize()};
   int N_pow2 {Pairwise::Spearman::nextPower2(N)};
   int K {_base->_maxClusters};

   _buffers.in_index = ::CUDA::Buffer<int2>(1 * kernelSize);
   _buffers.work_X = ::CUDA::Buffer<float2>(N * kernelSize);
   _buffers.work_N = ::CUDA::Buffer<int>(1 * kernelSize);
   _buffers.work_x = ::CUDA::Buffer<float>(N_pow2 * kernelSize);
   _buffers.work_y = ::CUDA::Buffer<float>(N_pow2 * kernelSize);
   _buffers.work_labels = ::CUDA::Buffer<qint8>(N * kernelSize);
   _buffers.work_components = ::CUDA::Buffer<cu_component>(K * kernelSize);
   _buffers.work_MP = ::CUDA::Buffer<float2>(K * kernelSize);
   _buffers.work_counts = ::CUDA::Buffer<int>(K * kernelSize);
   _buffers.work_logpi = ::CUDA::Buffer<float>(K * kernelSize);
   _buffers.work_loggamma = ::CUDA::Buffer<float>(N * K * kernelSize);
   _buffers.work_logGamma = ::CUDA::Buffer<float>(K * kernelSize);
   _buffers.work_rank = ::CUDA::Buffer<int>(N_pow2 * kernelSize);
   _buffers.out_K = ::CUDA::Buffer<qint8>(1 * kernelSize);
   _buffers.out_labels = ::CUDA::Buffer<qint8>(N * kernelSize);
   _buffers.out_correlations = ::CUDA::Buffer<float>(K * kernelSize);
}






std::unique_ptr<EAbstractAnalytic::Block> Similarity::CUDA::Worker::execute(const EAbstractAnalytic::Block* block)
{
   if ( ELog::isActive() )
   {
      ELog() << tr("Executing(CUDA) work index %1.\n").arg(block->index());
   }

   // cast block to work block
   const WorkBlock* workBlock {block->cast<const WorkBlock>()};

   // initialize result block
   ResultBlock* resultBlock {new ResultBlock(workBlock->index(), workBlock->start())};

   // bind cuda context to current thread
   _baseCuda->_context->setCurrent();

   // iterate through all pairs
   Pairwise::Index index {workBlock->start()};

   for ( int i = 0; i < workBlock->size(); i += _base->_kernelSize )
   {
      // write input buffers to device
      int steps {min(_base->_kernelSize, (int)workBlock->size() - i)};

      for ( int j = 0; j < steps; ++j )
      {
         _buffers.in_index[j] = { index.getX(), index.getY() };
         ++index;
      }

      for ( int j = steps; j < _base->_kernelSize; ++j )
      {
         _buffers.in_index[j] = { 0, 0 };
      }

      _buffers.in_index.write(_stream);

      // execute fetch-pair kernel
      _kernels.fetchPair.execute(
         _stream,
         _base->_kernelSize,
         &_baseCuda->_expressions,
         _base->_input->sampleSize(),
         &_buffers.in_index,
         _base->_minExpression,
         &_buffers.work_X,
         &_buffers.work_N,
         &_buffers.out_labels
      );

      // execute clustering kernel
      if ( _base->_clusMethod == ClusteringMethod::GMM )
      {
         _kernels.gmm.execute(
            _stream,
            _base->_kernelSize,
            _base->_input->sampleSize(),
            _base->_minSamples,
            _base->_minClusters,
            _base->_maxClusters,
            (int) _base->_criterion,
            _base->_removePreOutliers,
            _base->_removePostOutliers,
            &_buffers.work_X,
            &_buffers.work_N,
            &_buffers.work_x,
            &_buffers.work_y,
            &_buffers.work_labels,
            &_buffers.work_components,
            &_buffers.work_MP,
            &_buffers.work_counts,
            &_buffers.work_logpi,
            &_buffers.work_loggamma,
            &_buffers.work_logGamma,
            &_buffers.out_K,
            &_buffers.out_labels
         );
      }
      else
      {
         // set cluster size to 1 if clustering is disabled
         for ( int i = 0; i < _base->_kernelSize; ++i )
         {
            _buffers.out_K[i] = 1;
         }

         _buffers.out_K.write(_stream);
      }

      // execute correlation kernel
      if ( _base->_corrMethod == CorrelationMethod::Pearson )
      {
         _kernels.pearson.execute(
            _stream,
            _base->_kernelSize,
            &_buffers.work_X,
            _base->_maxClusters,
            &_buffers.out_labels,
            _base->_input->sampleSize(),
            _base->_minSamples,
            &_buffers.out_correlations
         );
      }
      else if ( _base->_corrMethod == CorrelationMethod::Spearman )
      {
         _kernels.spearman.execute(
            _stream,
            _base->_kernelSize,
            &_buffers.work_X,
            _base->_maxClusters,
            &_buffers.out_labels,
            _base->_input->sampleSize(),
            _base->_minSamples,
            &_buffers.work_x,
            &_buffers.work_y,
            &_buffers.work_rank,
            &_buffers.out_correlations
         );
      }

      // read results from device
      _buffers.out_K.read(_stream);
      _buffers.out_labels.read(_stream);
      _buffers.out_correlations.read(_stream);

      // wait for everything to finish
      _stream.wait();

      // save results
      for ( int j = 0; j < steps; ++j )
      {
         const qint8 *labels = &_buffers.out_labels.at(j * _base->_input->sampleSize());
         const float *correlations = &_buffers.out_correlations.at(j * _base->_maxClusters);

         Pair pair;
         pair.K = _buffers.out_K.at(j);

         if ( pair.K > 1 )
         {
            pair.labels = ResultBlock::makeVector(labels, _base->_input->sampleSize());
         }

         if ( pair.K > 0 )
         {
            pair.correlations = ResultBlock::makeVector(correlations, _base->_maxClusters);
         }

         resultBlock->append(pair);
      }
   }

   // return result block
   return unique_ptr<EAbstractAnalytic::Block>(resultBlock);
}
