#include "correlationmatrix.h"



using namespace std;
using namespace Pairwise;






QAbstractTableModel* CorrelationMatrix::model()
{
   return nullptr;
}






QVariant CorrelationMatrix::headerData(int section, Qt::Orientation orientation, int role) const
{
   // orientation is not used
   Q_UNUSED(orientation);

   // if role is not display return nothing
   if ( role != Qt::DisplayRole )
   {
      return QVariant();
   }

   // get genes metadata and make sure it is an array
   const EMetadata& genes {geneNames()};
   if ( genes.isArray() )
   {
      // make sure section is within limits of gene name array
      if ( section >= 0 && section < genes.toArray().size() )
      {
         // return gene name
         return genes.toArray().at(section).toString();
      }
   }

   // no gene found return nothing
   return QVariant();
}






QVariant CorrelationMatrix::data(const QModelIndex& index, int role) const
{
   // if role is not display return nothing
   if ( role != Qt::DisplayRole )
   {
      return QVariant();
   }

   // if row and column are equal return one
   if ( index.row() == index.column() )
   {
      return "1";
   }

   // get constant pair and read in values
   const Pair pair(this);
   int x {index.row()};
   int y {index.column()};
   if ( y > x )
   {
      swap(x,y);
   }
   pair.read({x,y});

   // Return value of pair as a string
   return pair.toString();
}






int CorrelationMatrix::rowCount(const QModelIndex&) const
{
   return geneSize();
}






int CorrelationMatrix::columnCount(const QModelIndex&) const
{
   return geneSize();
}






void CorrelationMatrix::initialize(const EMetadata &geneNames, int maxClusterSize, const EMetadata &correlationNames)
{
   // make sure correlation names is an array and is not empty
   if ( !correlationNames.isArray() || correlationNames.toArray().isEmpty() )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(tr("Domain Error"));
      e.setDetails(tr("Correlation names metadata is not an array or is empty."));
      throw e;
   }

   // save correlation names to metadata
   EMetaObject metaObject {meta().toObject()};
   metaObject.insert("correlations", correlationNames);
   setMeta(metaObject);

   // save correlation size and initialize base class
   _correlationSize = correlationNames.toArray().size();
   Matrix::initialize(geneNames, maxClusterSize, _correlationSize * sizeof(float), DATA_OFFSET);
}






EMetadata CorrelationMatrix::correlationNames() const
{
   return meta().toObject().at("correlations");
}






QVector<float> CorrelationMatrix::dumpRawData() const
{
   // if there are no genes do nothing
   if ( geneSize() == 0 )
   {
      return QVector<float>();
   }

   // create new correlation matrix
   QVector<float> data(geneSize() * geneSize() * maxClusterSize());

   // iterate through all pairs
   Pair pair(this);

   while ( pair.hasNext() )
   {
      // read in next pair
      pair.readNext();

      // load cluster data
      int i = pair.index().getX();
      int j = pair.index().getY();

      for ( int k = 0; k < pair.clusterSize(); ++k )
      {
         float correlation = pair.at(k, 0);

         data[i * geneSize() * maxClusterSize() + j * maxClusterSize() + k] = correlation;
         data[j * geneSize() * maxClusterSize() + i * maxClusterSize() + k] = correlation;
      }
   }

   return data;
}






void CorrelationMatrix::Pair::addCluster(int amount) const
{
   // keep adding a new list of floats for given amount
   while ( amount-- > 0 )
   {
      _correlations.append(QVector<float>(_cMatrix->_correlationSize, NAN));
   }
}






QString CorrelationMatrix::Pair::toString() const
{
   // if there are no correlations simply return null
   if ( _correlations.isEmpty() )
   {
      return tr("");
   }

   // initialize list of strings and iterate through all clusters
   QStringList ret;
   for (const auto& cluster : _correlations)
   {
      // initialize list of strings for cluster and iterate through each correlation
      QStringList clusterStrings;
      for (const auto& correlation : cluster)
      {
         // add correlation value as string
         clusterStrings << QString::number(correlation);
      }

      // join all cluster strings into one string
      ret << clusterStrings.join(',');
   }

   // join all clusters and return as string
   return ret.join(',');
}






void CorrelationMatrix::Pair::writeCluster(EDataStream& stream, int cluster)
{
   // make sure cluster value is within range
   if ( cluster >= 0 && cluster < _correlations.size() )
   {
      // write correlations per cluster to output stream
      for (const auto& correlation : _correlations.at(cluster))
      {
         stream << correlation;
      }
   }
}






void CorrelationMatrix::Pair::readCluster(const EDataStream& stream, int cluster) const
{
   // make sure cluster value is within range
   if ( cluster >= 0 && cluster < _correlations.size() )
   {
      // read correlations per cluster from input stream
      for (int i = 0; i < _cMatrix->_correlationSize ;++i)
      {
         float value;
         stream >> value;
         _correlations[cluster][i] = value;
      }
   }
}
