#ifndef HISTORY_H
#define HISTORY_H
#include <string>
#include "histitem.h"
#include "exception.h"



class History
{
public:
   // *
   // * DECLERATIONS
   // *
   class Iterator;
   using FPtr = FileMem::Ptr;
   // *
   // * BASIC METHODS
   // *
   History(const History&) = delete;
   History(History&&) = delete;
   History(FileMem&,FPtr = fNullPtr);
   // *
   // * FUNCTIONS
   // *
   inline FileMem::Ptr addr();
   void add_child(const History&);
   inline HistItem& head();
   inline Iterator begin();
   inline Iterator end();
   // *
   // * OPERATORS
   // *
   History& operator=(const History&) = delete;
   History& operator=(History&&) = delete;
   // *
   // * CONSTANTS
   // *
   const static auto fNullPtr = FileMem::nullPtr;
private:
   // *
   // * VARIABLES
   // *
   FileMem& _mem;
   HistItem _head;
};



class History::Iterator
{
public:
   // *
   // * DECLERATIONS
   // *
   friend class History;
   using FPtr = History::FPtr;
   // *
   // * FUNCTIONS
   // *
   inline Iterator childHead();
   // *
   // * OPERATORS
   // *
   inline HistItem& operator*();
   inline HistItem* operator->();
   inline void operator++();
   inline bool operator!=(const Iterator&);
private:
   // *
   // * BASIC METHODS
   // *
   inline Iterator(FileMem&,FPtr = fNullPtr);
   // *
   // * VARIABLES
   // *
   FileMem& _mem;
   HistItem _item;
};



inline History::History(FileMem& mem, FPtr ptr):
   _mem(mem),
   _head(mem,ptr)
{
   if (ptr==fNullPtr)
   {
      _head.allocate();
      _head.sync();
   }
}



inline FileMem::Ptr History::addr()
{
   return _head.addr();
}



inline HistItem& History::head()
{
   return _head;
}



inline History::Iterator History::begin()
{
   return {_mem,_head.childHead()};
}



inline History::Iterator History::end()
{
   return {_mem};
}



inline History::Iterator History::Iterator::childHead()
{
   return {_mem,_item.childHead()};
}



inline HistItem& History::Iterator::operator*()
{
   return _item;
}



inline HistItem* History::Iterator::operator->()
{
   return &_item;
}



inline void History::Iterator::operator++()
{
   if (_item.addr()!=FileMem::nullPtr)
   {
      _item = _item.next();
   }
}



inline bool History::Iterator::operator!=(const Iterator& cmp)
{
   return _item.addr()!=cmp._item.addr();
}



inline History::Iterator::Iterator(FileMem& mem, FPtr ptr):
   _mem(mem),
   _item(mem,ptr)
{}



#endif
