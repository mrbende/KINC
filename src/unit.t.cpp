#include <iostream>
#include "unit.h"
#include "console.h"
#include "linuxterm.h"



int main(int argc, char* argv[])
{
   if (unit::linuxfile::main())
   {
      std::cout << "ALL PASSED." << std::endl;
   }
   else
   {
      std::cout << "FAILURE." << std::endl;
      exit(1);
   }
   try
   {
      LinuxTerm::stty_raw();
      LinuxTerm terminal;
      Console console(argc,argv,terminal);
      console.run();
   }
   catch (...)
   {
      LinuxTerm::stty_cooked();
      throw;
   }
   LinuxTerm::stty_cooked();
   return 0;
}
