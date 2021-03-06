#include "similarity_input.h"
#include "datafactory.h"
#include "pairwise_gmm.h"
#include "pairwise_kmeans.h"
#include "pairwise_pearson.h"
#include "pairwise_spearman.h"






const QStringList Similarity::Input::CLUSTERING_NAMES
{
   "none"
   ,"gmm"
   ,"kmeans"
};






const QStringList Similarity::Input::CORRELATION_NAMES
{
   "pearson"
   ,"spearman"
};






const QStringList Similarity::Input::CRITERION_NAMES
{
   "BIC"
   ,"ICL"
};






Similarity::Input::Input(Similarity* parent):
   EAbstractAnalytic::Input(parent),
   _base(parent)
{
}






int Similarity::Input::size() const
{
   return Total;
}






EAbstractAnalytic::Input::Type Similarity::Input::type(int index) const
{
   switch (index)
   {
   case InputData: return Type::DataIn;
   case ClusterData: return Type::DataOut;
   case CorrelationData: return Type::DataOut;
   case ClusteringType: return Type::Selection;
   case CorrelationType: return Type::Selection;
   case MinExpression: return Type::Double;
   case MinSamples: return Type::Integer;
   case MinClusters: return Type::Integer;
   case MaxClusters: return Type::Integer;
   case CriterionType: return Type::Selection;
   case RemovePreOutliers: return Type::Boolean;
   case RemovePostOutliers: return Type::Boolean;
   case MinCorrelation: return Type::Double;
   case MaxCorrelation: return Type::Double;
   case KernelSize: return Type::Integer;
   default: return Type::Boolean;
   }
}






QVariant Similarity::Input::data(int index, Role role) const
{
   switch (index)
   {
   case InputData:
      switch (role)
      {
      case Role::CommandLineName: return QString("input");
      case Role::Title: return tr("Expression Matrix:");
      case Role::WhatsThis: return tr("Input expression matrix.");
      case Role::DataType: return DataFactory::ExpressionMatrixType;
      default: return QVariant();
      }
   case ClusterData:
      switch (role)
      {
      case Role::CommandLineName: return QString("ccm");
      case Role::Title: return tr("Cluster Matrix:");
      case Role::WhatsThis: return tr("Output matrix that will contain pairwise clusters.");
      case Role::DataType: return DataFactory::CCMatrixType;
      default: return QVariant();
      }
   case CorrelationData:
      switch (role)
      {
      case Role::CommandLineName: return QString("cmx");
      case Role::Title: return tr("Correlation Matrix:");
      case Role::WhatsThis: return tr("Output matrix that will contain pairwise correlations.");
      case Role::DataType: return DataFactory::CorrelationMatrixType;
      default: return QVariant();
      }
   case ClusteringType:
      switch (role)
      {
      case Role::CommandLineName: return QString("clusmethod");
      case Role::Title: return tr("Clustering Method:");
      case Role::WhatsThis: return tr("Method to use for pairwise clustering.");
      case Role::SelectionValues: return CLUSTERING_NAMES;
      case Role::Default: return "none";
      default: return QVariant();
      }
   case CorrelationType:
      switch (role)
      {
      case Role::CommandLineName: return QString("corrmethod");
      case Role::Title: return tr("Correlation Method:");
      case Role::WhatsThis: return tr("Method to use for pairwise correlation.");
      case Role::SelectionValues: return CORRELATION_NAMES;
      case Role::Default: return "pearson";
      default: return QVariant();
      }
   case MinExpression:
      switch (role)
      {
      case Role::CommandLineName: return QString("minexpr");
      case Role::Title: return tr("Minimum Expression:");
      case Role::WhatsThis: return tr("Minimum threshold for a sample to be included in a gene pair.");
      case Role::Default: return -std::numeric_limits<float>::infinity();
      case Role::Minimum: return -std::numeric_limits<float>::infinity();
      case Role::Maximum: return +std::numeric_limits<float>::infinity();
      default: return QVariant();
      }
   case MinSamples:
      switch (role)
      {
      case Role::CommandLineName: return QString("minsamp");
      case Role::Title: return tr("Minimum Sample Size:");
      case Role::WhatsThis: return tr("Minimum number of shared samples for a gene pair to be processed.");
      case Role::Default: return 30;
      case Role::Minimum: return 1;
      case Role::Maximum: return std::numeric_limits<int>::max();
      default: return QVariant();
      }
   case MinClusters:
      switch (role)
      {
      case Role::CommandLineName: return QString("minclus");
      case Role::Title: return tr("Minimum Clusters:");
      case Role::WhatsThis: return tr("Minimum number of clusters to test.");
      case Role::Default: return 1;
      case Role::Minimum: return 1;
      case Role::Maximum: return Pairwise::Index::MAX_CLUSTER_SIZE;
      default: return QVariant();
      }
   case MaxClusters:
      switch (role)
      {
      case Role::CommandLineName: return QString("maxclus");
      case Role::Title: return tr("Maximum Clusters:");
      case Role::WhatsThis: return tr("Maximum number of clusters to test.");
      case Role::Default: return 5;
      case Role::Minimum: return 1;
      case Role::Maximum: return Pairwise::Index::MAX_CLUSTER_SIZE;
      default: return QVariant();
      }
   case CriterionType:
      switch (role)
      {
      case Role::CommandLineName: return QString("crit");
      case Role::Title: return tr("Criterion:");
      case Role::WhatsThis: return tr("Criterion to select a clustering model.");
      case Role::SelectionValues: return CRITERION_NAMES;
      case Role::Default: return "ICL";
      default: return QVariant();
      }
   case RemovePreOutliers:
      switch (role)
      {
      case Role::CommandLineName: return QString("preout");
      case Role::Title: return tr("Remove pre-outliers:");
      case Role::WhatsThis: return tr("Whether to remove pre-clustering outliers.");
      case Role::Default: return false;
      default: return QVariant();
      }
   case RemovePostOutliers:
      switch (role)
      {
      case Role::CommandLineName: return QString("postout");
      case Role::Title: return tr("Remove post-outliers:");
      case Role::WhatsThis: return tr("Whether to remove post-clustering outliers.");
      case Role::Default: return false;
      default: return QVariant();
      }
   case MinCorrelation:
      switch (role)
      {
      case Role::CommandLineName: return QString("mincorr");
      case Role::Title: return tr("Minimum Correlation:");
      case Role::WhatsThis: return tr("Minimum threshold (absolute value) for a correlation to be saved.");
      case Role::Default: return 0.5;
      case Role::Minimum: return 0;
      case Role::Maximum: return 1;
      default: return QVariant();
      }
   case MaxCorrelation:
      switch (role)
      {
      case Role::CommandLineName: return QString("maxcorr");
      case Role::Title: return tr("Maximum Correlation:");
      case Role::WhatsThis: return tr("Maximum threshold (absolute value) for a correlation to be saved.");
      case Role::Default: return 1;
      case Role::Minimum: return 0;
      case Role::Maximum: return 1;
      default: return QVariant();
      }
   case KernelSize:
      switch (role)
      {
      case Role::CommandLineName: return QString("ksize");
      case Role::Title: return tr("Kernel Size:");
      case Role::WhatsThis: return tr("(OpenCL) Total number of kernels to run per block.");
      case Role::Default: return 4096;
      case Role::Minimum: return 1;
      case Role::Maximum: return std::numeric_limits<int>::max();
      default: return QVariant();
      }
   default: return QVariant();
   }
}






void Similarity::Input::set(int index, const QVariant& value)
{
   switch (index)
   {
   case ClusteringType:
      _base->_clusMethod = static_cast<ClusteringMethod>(CLUSTERING_NAMES.indexOf(value.toString()));

      switch ( _base->_clusMethod )
      {
      case ClusteringMethod::None:
         _base->_clusModel = nullptr;
         break;
      case ClusteringMethod::GMM:
         _base->_clusModel = new Pairwise::GMM();
         break;
      case ClusteringMethod::KMeans:
         _base->_clusModel = new Pairwise::KMeans();
         break;
      }
      break;
   case CorrelationType:
      _base->_corrMethod = static_cast<CorrelationMethod>(CORRELATION_NAMES.indexOf(value.toString()));

      switch ( _base->_corrMethod )
      {
      case CorrelationMethod::Pearson:
         _base->_corrModel = new Pairwise::Pearson();
         break;
      case CorrelationMethod::Spearman:
         _base->_corrModel = new Pairwise::Spearman();
         break;
      }
      break;
   case MinExpression:
      _base->_minExpression = value.toDouble();
      break;
   case MinSamples:
      _base->_minSamples = value.toInt();
      break;
   case MinClusters:
      _base->_minClusters = value.toInt();
      break;
   case MaxClusters:
      _base->_maxClusters = value.toInt();
      break;
   case CriterionType:
      _base->_criterion = static_cast<Pairwise::Criterion>(CRITERION_NAMES.indexOf(value.toString()));
      break;
   case RemovePreOutliers:
      _base->_removePreOutliers = value.toBool();
      break;
   case RemovePostOutliers:
      _base->_removePostOutliers = value.toBool();
      break;
   case MinCorrelation:
      _base->_minCorrelation = value.toDouble();
      break;
   case MaxCorrelation:
      _base->_maxCorrelation = value.toDouble();
      break;
   case KernelSize:
      _base->_kernelSize = value.toInt();
      break;
   }
}






void Similarity::Input::set(int index, QFile* file)
{
   Q_UNUSED(index)
   Q_UNUSED(file)
}






void Similarity::Input::set(int index, EAbstractData *data)
{
   switch (index)
   {
   case InputData:
      _base->_input = data->cast<ExpressionMatrix>();
      break;
   case ClusterData:
      _base->_ccm = data->cast<CCMatrix>();
      break;
   case CorrelationData:
      _base->_cmx = data->cast<CorrelationMatrix>();
      break;
   }
}
