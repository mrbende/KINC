# Include common settings
include (../KINC.pri)

# Basic Settings
TARGET = kinccore
TEMPLATE = lib
CONFIG += staticlib

# Build settings
DESTDIR = $$PWD/../../build/libs/

# Qt libraries
QT += core

# Preprocessor defines
DEFINES += QT_DEPRECATED_WARNINGS

# Used to ignore useless warnings from OpenCL
QMAKE_CXXFLAGS += -Wno-ignored-attributes

# Source files
SOURCES += \
   analyticfactory.cpp \
   ccmatrix_model.cpp \
   ccmatrix_pair.cpp \
   ccmatrix.cpp \
   correlationmatrix_model.cpp \
   correlationmatrix_pair.cpp \
   correlationmatrix.cpp \
   datafactory.cpp \
   exportcorrelationmatrix_input.cpp \
   exportcorrelationmatrix.cpp \
   exportexpressionmatrix_input.cpp \
   exportexpressionmatrix.cpp \
   expressionmatrix_gene.cpp \
   expressionmatrix_model.cpp \
   expressionmatrix.cpp \
   extract_input.cpp \
   extract.cpp \
   importcorrelationmatrix_input.cpp \
   importcorrelationmatrix.cpp \
   importexpressionmatrix_input.cpp \
   importexpressionmatrix.cpp \
   pairwise_clusteringmodel.cpp \
   pairwise_correlationmodel.cpp \
   pairwise_gmm.cpp \
   pairwise_index.cpp \
   pairwise_linalg.cpp \
   pairwise_matrix_pair.cpp \
   pairwise_matrix.cpp \
   pairwise_pearson.cpp \
   pairwise_spearman.cpp \
   powerlaw_input.cpp \
   powerlaw.cpp \
   rmt_input.cpp \
   rmt.cpp \
   similarity_cuda_fetchpair.cpp \
   similarity_cuda_gmm.cpp \
   similarity_cuda_pearson.cpp \
   similarity_cuda_spearman.cpp \
   similarity_cuda_worker.cpp \
   similarity_cuda.cpp \
   similarity_input.cpp \
   similarity_opencl_fetchpair.cpp \
   similarity_opencl_gmm.cpp \
   similarity_opencl_pearson.cpp \
   similarity_opencl_spearman.cpp \
   similarity_opencl_worker.cpp \
   similarity_opencl.cpp \
   similarity_resultblock.cpp \
   similarity_serial.cpp \
   similarity_workblock.cpp \
   similarity.cpp

# Header files
HEADERS += \
   analyticfactory.h \
   ccmatrix_model.h \
   ccmatrix_pair.h \
   ccmatrix.h \
   correlationmatrix_model.h \
   correlationmatrix_pair.h \
   correlationmatrix.h \
   datafactory.h \
   exportcorrelationmatrix_input.h \
   exportcorrelationmatrix.h \
   exportexpressionmatrix_input.h \
   exportexpressionmatrix.h \
   expressionmatrix_gene.h \
   expressionmatrix_model.h \
   expressionmatrix.h \
   extract_input.h \
   extract.h \
   importcorrelationmatrix_input.h \
   importcorrelationmatrix.h \
   importexpressionmatrix_input.h \
   importexpressionmatrix.h \
   pairwise_clusteringmodel.h \
   pairwise_correlationmodel.h \
   pairwise_gmm.h \
   pairwise_index.h \
   pairwise_linalg.h \
   pairwise_matrix_pair.h \
   pairwise_matrix.h \
   pairwise_pearson.h \
   pairwise_spearman.h \
   powerlaw_input.h \
   powerlaw.h \
   rmt_input.h \
   rmt.h \
   similarity_cuda_fetchpair.h \
   similarity_cuda_gmm.h \
   similarity_cuda_pearson.h \
   similarity_cuda_spearman.h \
   similarity_cuda_worker.h \
   similarity_cuda.h \
   similarity_input.h \
   similarity_opencl_fetchpair.h \
   similarity_opencl_gmm.h \
   similarity_opencl_pearson.h \
   similarity_opencl_spearman.h \
   similarity_opencl_worker.h \
   similarity_opencl.h \
   similarity_resultblock.h \
   similarity_serial.h \
   similarity_workblock.h \
   similarity.h
