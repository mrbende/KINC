#ifndef DATAMAP_H
#define DATAMAP_H
#include <unordered_map>
#include <string>
#include "dataplugin.h"



class DataMap
{
public:
   using StdMap = std::unordered_map<std::string,DataPlugin*>;
   // *
   // * BASIC METHODS
   // *
   DataMap(const DataMap&) = delete;
   DataMap(DataMap&&) = delete;
   DataMap& operator=(const DataMap&) = delete;
   DataMap& operator=(DataMap&&) = delete;
   DataMap() = default;
   // *
   // * FUNCTIONS
   // *
   bool add(const std::string&,DataPlugin*);
   bool del(const std::string&);
   bool del(DataPlugin*);
   StdMap::iterator find(const std::string&);
   StdMap::iterator begin();
   StdMap::iterator end();
private:
   // *
   // * VARIABLES
   // *
   StdMap _map;
};



#endif
