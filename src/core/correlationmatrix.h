#ifndef CORRELATIONMATRIX_H
#define CORRELATIONMATRIX_H
#include "pairwise_matrix.h"



class CorrelationMatrix : public Pairwise::Matrix
{
   Q_OBJECT
public:
   class Pair;
   virtual QAbstractTableModel* model() override final;
   QVariant headerData(int section, Qt::Orientation orientation, int role) const;
   int rowCount(const QModelIndex&) const;
   int columnCount(const QModelIndex&) const;
   QVariant data(const QModelIndex& index, int role) const;
   void initialize(const EMetadata& geneNames, int maxClusterSize, const EMetadata& correlationNames);
   EMetadata correlationNames() const;
   QVector<float> dumpRawData() const;
private:
   virtual void writeHeader() { stream() << _correlationSize; }
   virtual void readHeader() { stream() >> _correlationSize; }
   static const int DATA_OFFSET {1};
   qint8 _correlationSize {0};
};



class CorrelationMatrix::Pair : public Pairwise::Matrix::Pair
{
public:
   Pair(CorrelationMatrix* matrix):
      Matrix::Pair(matrix),
      _cMatrix(matrix)
      {}
   Pair(const CorrelationMatrix* matrix):
      Matrix::Pair(matrix),
      _cMatrix(matrix)
      {}
   Pair() = default;
   virtual void clearClusters() const { _correlations.clear(); }
   virtual void addCluster(int amount = 1) const;
   virtual int clusterSize() const { return _correlations.size(); }
   virtual bool isEmpty() const { return _correlations.isEmpty(); }
   QString toString() const;
   const float& at(int cluster, int correlation) const
      { return _correlations.at(cluster).at(correlation); }
   float& at(int cluster, int correlation) { return _correlations[cluster][correlation]; }
private:
   virtual void writeCluster(EDataStream& stream, int cluster);
   virtual void readCluster(const EDataStream& stream, int cluster) const;
   mutable QVector<QVector<float>> _correlations;
   const CorrelationMatrix* _cMatrix;
};



#endif
