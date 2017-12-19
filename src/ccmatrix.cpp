#include "ccmatrix.h"



using namespace std;
using namespace GenePair;






void CCMatrix::Pair::addCluster(int amount) const
{
   // keep adding a new list of sample masks for given amount
   while ( amount-- > 0 )
   {
      _sampleMasks.push_back({});
      for (int i = 0; i < _cMatrix->_sampleSize ;++i)
      {
         _sampleMasks.back().push_back(0);
      }
   }
}






QString CCMatrix::Pair::toString() const
{
   // if there are no clusters simply return null type
   if ( _sampleMasks.isEmpty() )
   {
      QString ret("(");
      for (int i = 0; i < _cMatrix->_sampleSize ;++i)
      {
         ret.append("1");
      }
      ret.append(")");
      return ret;
   }

   // initialize list of strings and iterate through all clusters
   QStringList ret;
   for (const auto& cluster : _sampleMasks)
   {
      // initialize list of strings for cluster and iterate through each sample
      QString clusterString("(");
      for (const auto& sample : cluster)
      {
         // add new sample token as hexadecimal allowing 16 different possible values
         switch (sample)
         {
         case 0:
         case 1:
         case 2:
         case 3:
         case 4:
         case 5:
         case 6:
         case 7:
         case 8:
         case 9:
            clusterString.append(QString::number(sample));
            break;
         case 10:
            clusterString.append("A");
            break;
         case 11:
            clusterString.append("B");
            break;
         case 12:
            clusterString.append("C");
            break;
         case 13:
            clusterString.append("D");
            break;
         case 14:
            clusterString.append("E");
            break;
         case 15:
            clusterString.append("F");
            break;
         }
      }

      // join all cluster string into one string
      ret << clusterString.append(')');
   }

   // join all clusters and return as string
   return ret.join(',');
}






void CCMatrix::Pair::writeCluster(EDataStream &stream, int cluster)
{
   // make sure cluster value is within range
   if ( cluster >= 0 && cluster < _sampleMasks.size() )
   {
      // iterate through each correlation and write to object
      for (const auto& sample : _sampleMasks.at(cluster))
      {
         stream << sample;
      }
   }
}






void CCMatrix::Pair::readCluster(const EDataStream &stream, int cluster) const
{
   // make sure cluster value is within range
   if ( cluster >= 0 && cluster < _sampleMasks.size() )
   {
      // clear cluster and reserve for given size
      _sampleMasks[cluster].clear();
      _sampleMasks[cluster].reserve(_cMatrix->_sampleSize);

      // read in number of samples per cluster and add to list
      for (int i = 0; i < _cMatrix->_sampleSize ;++i)
      {
         qint8 value;
         stream >> value;
         _sampleMasks[cluster].push_back(value);
      }
   }
}






QVariant CCMatrix::headerData(int section, Qt::Orientation orientation, int role) const
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
      if ( section >= 0 && section < genes.toArray()->size() )
      {
         // return gene name
         return genes.toArray()->at(section)->toVariant();
      }
   }

   // no gene found return nothing
   return QVariant();
}






QVariant CCMatrix::data(const QModelIndex &index, int role) const
{
   // if role is not display return nothing
   if ( role != Qt::DisplayRole )
   {
      return QVariant();
   }

   // if row and column are equal return one
   if ( index.row() == index.column() )
   {
      return "(1)";
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

   // Return value of gene pair as a string
   return pair.toString();
}






void CCMatrix::initialize(const EMetadata &geneNames, const EMetadata &sampleNames)
{
   // make sure sample names is an array and is not empty
   if ( !sampleNames.isArray() || sampleNames.toArray()->isEmpty() )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(tr("Domain Error"));
      e.setDetails(tr("Correlation names metadata is not an array or empty."));
      throw e;
   }

   // get map of metadata root and make copy of sample names
   EMetadata::Map* map {meta().toObject()};
   map->insert("samples",new EMetadata(sampleNames));

   // save sample size and initialize base class
   _sampleSize = sampleNames.toArray()->size();
   Base::initialize(geneNames,_sampleSize,DATA_OFFSET);
}






const EMetadata &CCMatrix::sampleNames() const
{
   // get metadata root and make sure samples key exist
   const EMetadata::Map* map {meta().toObject()};
   if ( !map->contains("samples") )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(QObject::tr("Null Return Reference"));
      e.setDetails(QObject::tr("Requesting reference to sample names when none exists."));
      throw e;
   }

   // return correlation names list
   return *(*map)["samples"];
}
