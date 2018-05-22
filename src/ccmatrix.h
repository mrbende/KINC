#ifndef CCMATRIX_H
#define CCMATRIX_H
#include "pairwise_matrix.h"



class CCMatrix : public QAbstractTableModel, public Pairwise::Matrix
{
   Q_OBJECT
public:
   class Pair : public Matrix::Pair
   {
   public:
      Pair(CCMatrix* matrix):
         Matrix::Pair(matrix),
         _cMatrix(matrix)
         {}
      Pair(const CCMatrix* matrix):
         Matrix::Pair(matrix),
         _cMatrix(matrix)
         {}
      Pair() {}
      virtual void clearClusters() const { _sampleMasks.clear(); }
      virtual void addCluster(int amount = 1) const;
      virtual int clusterSize() const { return _sampleMasks.size(); }
      virtual bool isEmpty() const { return _sampleMasks.isEmpty(); }
      QString toString() const;
      const qint8& at(int cluster, int sample) const { return _sampleMasks.at(cluster).at(sample); }
      qint8& at(int cluster, int sample) { return _sampleMasks[cluster][sample]; }
   private:
      virtual void writeCluster(EDataStream& stream, int cluster);
      virtual void readCluster(const EDataStream& stream, int cluster) const;
      mutable QList<QList<qint8>> _sampleMasks;
      const CCMatrix* _cMatrix;
   };
   virtual QAbstractTableModel* getModel() override final { return this; }
   virtual QVariant headerData(int section, Qt::Orientation orientation, int role) const;
   virtual int rowCount(const QModelIndex&) const override final { return geneSize(); }
   virtual int columnCount(const QModelIndex&) const override final { return geneSize(); }
   virtual QVariant data(const QModelIndex& index, int role) const override final;
   void initialize(const EMetadata& geneNames, const EMetadata& sampleNames);
   const EMetadata& sampleNames() const;
   int sampleSize() const { return _sampleSize; }
private:
   virtual void writeHeader() { stream() << _sampleSize; }
   virtual void readHeader() { stream() >> _sampleSize; }
   static const int DATA_OFFSET {4};
   qint32 _sampleSize {0};
};



#endif
