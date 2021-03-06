#include "pairwise_index.h"




using namespace Pairwise;






Index::Index(qint32 x, qint32 y):
   _x(x),
   _y(y)
{
   // make sure pairwise index is valid
   if ( x < 1 || y < 0 || x <= y )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(QObject::tr("Pairwise Index Error"));
      e.setDetails(QObject::tr("Cannot initialize pairwise index (%1, %2).").arg(x).arg(y));
      throw e;
   }
}






Index::Index(qint64 index):
   _x(1),
   _y(0)
{
   // make sure index is valid
   if ( index < 0 )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(QObject::tr("Pairwise Index Error"));
      e.setDetails(QObject::tr("Cannot initialize pairwise index from %1.").arg(index));
      throw e;
   }

   // compute pairwise index from scalar index
   qint64 pos {0};
   while ( pos <= index )
   {
      ++_x;
      pos = _x * (_x - 1) / 2;
   }

   --_x;
   pos = _x * (_x - 1) / 2;

   _y = index - pos;
}






qint64 Index::indent(qint8 cluster) const
{
   // make sure cluster given is valid
   if ( cluster < 0 || cluster >= MAX_CLUSTER_SIZE )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(QObject::tr("Pairwise Index Error"));
      e.setDetails(QObject::tr("Cluster %1 is outside limits.").arg(cluster));
      throw e;
   }

   // compute indent with given cluster and return it
   qint64 index {(qint64)_x * (_x - 1) / 2 + _y};
   return index * MAX_CLUSTER_SIZE + cluster;
}






void Index::operator++()
{
   // increment gene y and check if it reaches gene x
   if ( ++_y >= _x )
   {
      // reset gene y to 0 and increment gene x
      _y = 0;
      ++_x;
   }
}






Index Index::operator++(int)
{
   // save index value, increment it, and return previous value
   Index ret {*this};
   ++(*this);
   return ret;
}
