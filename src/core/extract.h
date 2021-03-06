#ifndef EXTRACT_H
#define EXTRACT_H
#include <ace/core/core.h>

#include "ccmatrix.h"
#include "correlationmatrix.h"
#include "expressionmatrix.h"



class Extract : public EAbstractAnalytic
{
   Q_OBJECT
public:
   class Input;
   virtual int size() const override final;
   virtual void process(const EAbstractAnalytic::Block* result) override final;
   virtual EAbstractAnalytic::Input* makeInput() override final;
   virtual void initialize();
private:
   ExpressionMatrix* _emx {nullptr};
   CCMatrix* _ccm {nullptr};
   CorrelationMatrix* _cmx {nullptr};
   QFile* _output {nullptr};
   QFile* _graphml {nullptr};
   float _minCorrelation {0.85};
   float _maxCorrelation {1.00};
};



#endif
