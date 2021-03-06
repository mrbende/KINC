#ifndef PAIRWISE_KMEANS_H
#define PAIRWISE_KMEANS_H
#include "pairwise_clustering.h"

namespace Pairwise
{
   class KMeans : public Clustering
   {
   public:
      KMeans() = default;

   protected:
      bool fit(const QVector<Vector2>& X, int N, int K, QVector<qint8>& labels);

      float logLikelihood() const { return _logL; }
      float entropy() const { return 0; }

   private:
      float computeLogLikelihood(const QVector<Vector2>& X, int N, const QVector<qint8>& y);

      QVector<Vector2> _means;
      float _logL;
   };
}

#endif
