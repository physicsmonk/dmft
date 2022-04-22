//
//  input.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef input_hpp
#define input_hpp

#include <sstream>
#include <cstring>
#include <vector>
#include "pugixml.hpp"
#include <mpi.h>
#include <stdint.h>
#include <limits.h>


// Get underlying data type of std::size_t
#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "what is happening here?"
#endif


// Read from a xml doc root the data located at path. Use
// template function to handle various output datatypes
// Note docroot is passed by value as it will be changed in this function.
template <typename SomeType>
bool readxml(SomeType& data, pugi::xml_node docroot, const std::string& path)
{
  std::vector<std::string> pathlist;
  std::istringstream pathstream(path);
  std::string s;
  std::size_t l = 0;
  std::stringstream ss;

  // Split the path nodes for later use
  while (std::getline(pathstream, s, '/'))
  {
    pathlist.push_back(s);
  }
  // Test
  // for (const auto& i : pathlist) std::cout << i << "-->";
  // std::cout << std::endl;

  for (std::size_t i=0; i < pathlist.size()-1; ++i)
  {
    docroot = docroot.child(pathlist[i].c_str());
    l += pathlist[i].length();
    if (!docroot)
    {
      std::cout << "No " << path.substr(0, l+i)
      << " specified, using default value(s)" << std::endl;
      return false;
    }
  }
  // Test
  // std::cout << l << std::endl;

  l = pathlist.back().find(".");
  if (l != std::string::npos)
  {
    docroot = docroot.child(pathlist.back().substr(0, l).c_str());
    if (!docroot)
    {
      std::cout << "No "
      << path.substr(0, path.length()-(pathlist.back().length()-l))
      << " specified, using default value(s)" << std::endl;
      return false;
    }
    else if (docroot.attribute(pathlist.back().substr(l+1).c_str()))
    {
      ss << docroot.attribute(pathlist.back().substr(l+1).c_str()).value();
      ss >> data;
      std::cout << "Input " << path << " is " << data << std::endl;
    }
    else
    {
      std::cout << "No " << path << " specified, using default " << data << std::endl;
      return false;
    }
  }
  else
  {
    docroot = docroot.child(pathlist.back().c_str());
    if (!docroot)
    {
      std::cout << "No " << path << " specified, using default " << data << std::endl;
      return false;
    }
    else
    {
      ss << docroot.child_value();
      ss >> data;
      std::cout << "Input " << path << " is " << data << std::endl;
    }
  }

  return true;
}

// Overloaded data-reading-broadcasting function to handle different MPI datatypes and dolfin Constant object
bool readxml_bcast(std::string& data, const pugi::xml_node& docroot, const std::string& path, const MPI_Comm& comm, const int rank)  // String data
{
  bool readed;
  int l;

  if (rank == 0)
  {
    readed = readxml(data, docroot, path);
    if (readed) l = static_cast<int>(data.length());
  }
  MPI_Bcast(&readed, 1, MPI_CXX_BOOL, 0, comm);

  if (readed)
  {
    char* tmp = NULL;
    MPI_Bcast(&l, 1, MPI_INT, 0, comm);
    tmp = (char *) malloc((l+1)*sizeof(char));
    if (!tmp)
    {
      std::cout << "Error: Memory allocation for char* tmp failed (inside function readxml_bcast)" << std::endl;
      exit(1);
    }
    if (rank == 0) strcpy(tmp, data.c_str());  // Copy data string on root processor to character array for broadcasting
    MPI_Bcast(tmp, l+1, MPI_CHAR, 0, comm);
    data = tmp;  // Copy tmp back to data on all processors
    if (rank == 0) std::cout << "Broadcasted " << path << std::endl;
    free(tmp);
  }
  else
  {
    return false;
  }

  return true;
}

bool readxml_bcast(int& data, const pugi::xml_node& docroot, const std::string& path, const MPI_Comm& comm, const int rank)  // Integer data
{
    bool readed;

    if (rank == 0) readed = readxml(data, docroot, path);
    MPI_Bcast(&readed, 1, MPI_CXX_BOOL, 0, comm);

    if (readed)
    {
        MPI_Bcast(&data, 1, MPI_INT, 0, comm);
        if (rank == 0) std::cout << "Broadcasted " << path << std::endl;
    }
    else
    {
        return false;
    }

    return true;
}

bool readxml_bcast(bool& data, const pugi::xml_node& docroot, const std::string& path, const MPI_Comm& comm, const int rank)  // Integer data
{
    bool readed;

    if (rank == 0) readed = readxml(data, docroot, path);
    MPI_Bcast(&readed, 1, MPI_CXX_BOOL, 0, comm);

    if (readed)
    {
        MPI_Bcast(&data, 1, MPI_CXX_BOOL, 0, comm);
        if (rank == 0) std::cout << "Broadcasted " << path << std::endl;
    }
    else
    {
        return false;
    }

    return true;
}

bool readxml_bcast(std::size_t& data, const pugi::xml_node& docroot, const std::string& path, const MPI_Comm& comm, const int rank)  // Integer data
{
    bool readed;

    if (rank == 0) readed = readxml(data, docroot, path);
    MPI_Bcast(&readed, 1, MPI_CXX_BOOL, 0, comm);

    if (readed)
    {
        MPI_Bcast(&data, 1, my_MPI_SIZE_T, 0, comm);
        if (rank == 0) std::cout << "Broadcasted " << path << std::endl;
    }
    else
    {
        return false;
    }

    return true;
}

bool readxml_bcast(double& data, const pugi::xml_node& docroot, const std::string& path, const MPI_Comm& comm, const int rank, const double unit=-1.0)  // Double float data
{
    bool readed;

    if (rank == 0) readed = readxml(data, docroot, path);
    MPI_Bcast(&readed, 1, MPI_CXX_BOOL, 0, comm);

    if (readed)
    {
        MPI_Bcast(&data, 1, MPI_DOUBLE, 0, comm);
        if (unit > 0) data /= unit;
        if (rank == 0) std::cout << "Broadcasted " << path << std::endl;
    }
    else
    {
        return false;
    }

    return true;
}

#endif /* input_hpp */
