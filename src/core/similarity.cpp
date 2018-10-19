#include "similarity.h"
#include "similarity_input.h"
#include "similarity_resultblock.h"
#include "similarity_serial.h"
#include "similarity_workblock.h"
#include "similarity_opencl.h"
#include "similarity_cuda.h"
#include "ccmatrix_pair.h"
#include "correlationmatrix_pair.h"
#include <ace/core/elog.h>



using namespace std;






/*!
 * Return the total number of pairs that must be processed for a given
 * expression matrix.
 *
 * @param emx
 */
qint64 Similarity::totalPairs(const ExpressionMatrix* emx) const
{
   EDEBUG_FUNC(this,emx);

   return (qint64) emx->geneSize() * (emx->geneSize() - 1) / 2;
}






/*!
 * Return the total number of work blocks this analytic must process.
 */
int Similarity::size() const
{
   EDEBUG_FUNC(this);

   return (totalPairs(_input) + _workBlockSize - 1) / _workBlockSize;
}






/*!
 * Create and return a work block for this analytic with the given index. This
 * implementation creates a work block with a start index and size denoting the
 * number of pairs to process.
 *
 * @param index
 */
std::unique_ptr<EAbstractAnalytic::Block> Similarity::makeWork(int index) const
{
   EDEBUG_FUNC(this,index);

   if ( ELog::isActive() )
   {
      ELog() << tr("Making work index %1 of %2.\n").arg(index).arg(size());
   }

   qint64 start {index * _workBlockSize};
   qint64 size {min(totalPairs(_input) - start, (qint64) _workBlockSize)};

   return unique_ptr<EAbstractAnalytic::Block>(new WorkBlock(index, start, size));
}






/*!
 * Create an empty and uninitialized work block.
 */
std::unique_ptr<EAbstractAnalytic::Block> Similarity::makeWork() const
{
   EDEBUG_FUNC(this);

   return unique_ptr<EAbstractAnalytic::Block>(new WorkBlock);
}






/*!
 * Create an empty and uninitialized result block.
 */
std::unique_ptr<EAbstractAnalytic::Block> Similarity::makeResult() const
{
   EDEBUG_FUNC(this);

   return unique_ptr<EAbstractAnalytic::Block>(new ResultBlock);
}






/*!
 * Read in a block of results made from a block of work with the corresponding
 * index. This implementation takes the Pair objects in the result block and
 * saves them to the output correlation matrix and cluster matrix.
 *
 * @param result
 */
void Similarity::process(const EAbstractAnalytic::Block* result)
{
   EDEBUG_FUNC(this,result);

   if ( ELog::isActive() )
   {
      ELog() << tr("Processing result %1 of %2.\n").arg(result->index()).arg(size());
   }

   const ResultBlock* resultBlock {result->cast<ResultBlock>()};

   // iterate through all pairs in result block
   Pairwise::Index index {resultBlock->start()};

   for ( auto& pair : resultBlock->pairs() )
   {
      // save clusters whose correlations are within thresholds
      if ( pair.K > 1 )
      {
         CCMatrix::Pair ccmPair(_ccm);

         for ( qint8 k = 0; k < pair.K; ++k )
         {
            float corr = pair.correlations[k];

            if ( !isnan(corr) && _minCorrelation <= abs(corr) && abs(corr) <= _maxCorrelation )
            {
               ccmPair.addCluster();

               for ( int i = 0; i < _input->sampleSize(); ++i )
               {
                  ccmPair.at(ccmPair.clusterSize() - 1, i) = (pair.labels[i] >= 0)
                     ? (k == pair.labels[i])
                     : -pair.labels[i];
               }
            }
         }

         if ( ccmPair.clusterSize() > 0 )
         {
            ccmPair.write(index);
         }
      }

      // save correlations that are within thresholds
      if ( pair.K > 0 )
      {
         CorrelationMatrix::Pair cmxPair(_cmx);

         for ( qint8 k = 0; k < pair.K; ++k )
         {
            float corr = pair.correlations[k];

            if ( !isnan(corr) && _minCorrelation <= abs(corr) && abs(corr) <= _maxCorrelation )
            {
               cmxPair.addCluster();
               cmxPair.at(cmxPair.clusterSize() - 1, 0) = corr;
            }
         }

         if ( cmxPair.clusterSize() > 0 )
         {
            cmxPair.write(index);
         }
      }

      ++index;
   }
}






/*!
 * Make a new input object and return its pointer.
 */
EAbstractAnalytic::Input* Similarity::makeInput()
{
   EDEBUG_FUNC(this);

   return new Input(this);
}






/*!
 * Make a new serial object and return its pointer.
 */
EAbstractAnalytic::Serial* Similarity::makeSerial()
{
   EDEBUG_FUNC(this);

   return new Serial(this);
}






/*!
 * Make a new OpenCL object and return its pointer.
 */
EAbstractAnalytic::OpenCL* Similarity::makeOpenCL()
{
   EDEBUG_FUNC(this);

   return new OpenCL(this);
}






/*!
 * Make a new CUDA object and return its pointer.
 */
EAbstractAnalytic::CUDA* Similarity::makeCUDA()
{
   return new CUDA(this);
}






/*!
 * Initialize this analytic. This implementation checks to make sure that valid
 * arguments were provided.
 */
void Similarity::initialize()
{
   EDEBUG_FUNC(this);

   // only the master process needs to validate arguments
   if ( !isMaster() )
   {
      return;
   }

   // make sure input data is valid
   if ( !_input )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(tr("Invalid Argument"));
      e.setDetails(tr("Did not get a valid input data object."));
      throw e;
   }

   // make sure cluster range is valid
   if ( _maxClusters < _minClusters )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(tr("Invalid Argument"));
      e.setDetails(tr("Minimum clusters must be less than or equal to maximum clusters."));
      throw e;
   }
}






/*!
 * Initialize the output data objects of this analytic.
 */
void Similarity::initializeOutputs()
{
   EDEBUG_FUNC(this);

   // make sure output data is valid
   if ( !_ccm || !_cmx )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(tr("Invalid Argument"));
      e.setDetails(tr("Did not get valid output data objects."));
      throw e;
   }

   // initialize cluster matrix
   _ccm->initialize(_input->geneNames(), _maxClusters, _input->sampleNames());

   // initialize correlation matrix
   EMetaArray correlations;
   correlations.append(_corrName);

   _cmx->initialize(_input->geneNames(), _maxClusters, correlations);
}
