#include <ace/core/core.h>
#include <ace/core/ace_dataobject.h>

#include "testcorrelationmatrix.h"
#include "correlationmatrix.h"
#include "datafactory.h"



void TestCorrelationMatrix::test()
{
	// create random correlation data
	int numGenes = 10;
	int maxClusters = 5;
	QVector<Pair> testPairs;

	for ( int i = 0; i < numGenes; ++i )
	{
		for ( int j = 0; j < i; ++j )
		{
			int numClusters = rand() % (maxClusters + 1);

			if ( numClusters > 0 )
			{
				QVector<float> correlations(numClusters);

				for ( int k = 0; k < numClusters; ++k )
				{
					correlations[k] = -1.0 + 2.0 * rand() / (1 << 31);
				}

				testPairs.append({ { i, j }, correlations });
			}
		}
	}

	// create metadata
	EMetaArray metaGeneNames;
	for ( int i = 0; i < numGenes; ++i )
	{
		metaGeneNames.append(QString::number(i));
	}

	EMetaArray metaCorrelationNames;
	metaCorrelationNames.append(QString("test"));

	// create data object
	QString path {QDir::tempPath() + "/test.cmx"};

	std::unique_ptr<Ace::DataObject> dataRef {new Ace::DataObject(path, DataFactory::CorrelationMatrixType, EMetadata(EMetadata::Object))};
	CorrelationMatrix* matrix {dataRef->data()->cast<CorrelationMatrix>()};

	// write data to file
	matrix->initialize(metaGeneNames, maxClusters, metaCorrelationNames);

	CorrelationMatrix::Pair pair(matrix);

	for ( auto& testPair : testPairs )
	{
		pair.clearClusters();
		pair.addCluster(testPair.correlations.size());

		for ( int k = 0; k < pair.clusterSize(); ++k )
		{
			pair.at(k, 0) = testPair.correlations.at(k);
		}

		pair.write(testPair.index);
	}

	matrix->finish();

	// read and verify correlation data from file
	pair.reset();

	for ( auto& testPair : testPairs )
	{
		QVERIFY(pair.hasNext());
		pair.readNext();

		QCOMPARE(pair.index(), testPair.index);
		QCOMPARE(pair.clusterSize(), testPair.correlations.size());

		for ( int k = 0; k < pair.clusterSize(); ++k )
		{
			QCOMPARE(pair.at(k, 0), testPair.correlations.at(k));
		}
	}
}
