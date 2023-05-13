//
//  input.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef input_hpp
#define input_hpp

#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include "pugixml.hpp"
#include <mpi.h>

// Get underlying data type of Eigen::Index
//#include <cstdint>
//#include <climits>
//#if SIZE_MAX == UCHAR_MAX
//#define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
//#elif SIZE_MAX == USHRT_MAX
//#define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
//#elif SIZE_MAX == UINT_MAX
//#define my_MPI_SIZE_T MPI_UNSIGNED
//#elif SIZE_MAX == ULONG_MAX
//#define my_MPI_SIZE_T MPI_UNSIGNED_LONG
//#elif SIZE_MAX == ULLONG_MAX
//#define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
//#else
//#error "What is happening here?"
//#endif

// Read from a xml doc root the data located at path. Use
// template function to handle various output datatypes
// Note docroot is passed by value as it will be changed in this function.
template <typename T>
bool readxml(T& data, pugi::xml_node docroot, const std::string& path)
{
    std::vector<std::string> pathlist;
    std::istringstream pathstream(path);
    std::string s, text;
    int l = 0;
    
    // Split the path nodes for later use
    while (std::getline(pathstream, s, '/'))
    {
        pathlist.push_back(s);
    }
    // Test
    // for (const auto& i : pathlist) std::cout << i << "-->";
    // std::cout << std::endl;
    
    for (int i = 0; i + 1 < pathlist.size(); ++i)
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
        pugi::xml_attribute attr;
        docroot = docroot.child(pathlist.back().substr(0, l).c_str());
        if (!docroot)
        {
            std::cout << "No "
            << path.substr(0, path.length()-(pathlist.back().length()-l))
            << " specified, using default value(s)" << std::endl;
            return false;
        }
        else if ((attr = docroot.attribute(pathlist.back().substr(l+1).c_str())))
        {
            text = attr.value();
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
            text = docroot.child_value();
        }
    }
    
    if constexpr (std::is_same<T, std::string>::value) data = text;
    else if (std::is_same<T, int>::value || std::is_same<T, bool>::value) data = std::stoi(text);
    else if (std::is_same<T, std::ptrdiff_t>::value) {
#if PTRDIFF_MAX == INT_MAX
        data = std::stoi(text);
#elif PTRDIFF_MAX == LONG_MAX
        data = std::stol(text);
#elif PTRDIFF_MAX == LLONG_MAX
        data = std::stoll(text);
#else
#error "PTRDIFF_MAX is not INT_MAX, LONG_MAX, or LLONG_MAX"
#endif
    }
    else if (std::is_same<T, double>::value) data = std::stod(text);
    else if (std::is_same<T, float>::value) data = std::stof(text);
    else throw std::invalid_argument("readxml: input data type not implemented currently");
    
    std::cout << "Input " << path << " is " << data << std::endl;
    
    return true;
}

// Data-reading-broadcasting function
template <typename T>
bool readxml_bcast(T& data, const pugi::xml_node& docroot, const std::string& path, const MPI_Comm& comm, const double unit=-1.0)  // Integer data
{
    bool readed;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) readed = readxml(data, docroot, path);
    MPI_Bcast(&readed, 1, MPI_CXX_BOOL, 0, comm);
    
    if (readed)
    {
        if constexpr (std::is_same<T, std::string>::value) {
            int l;
            if (rank == 0) l = static_cast<int>(data.length());
            MPI_Bcast(&l, 1, MPI_INT, 0, comm);
            char *tmp = new char[l + 1];
            if (rank == 0) strcpy(tmp, data.c_str());
            MPI_Bcast(tmp, l + 1, MPI_CHAR, 0, comm);
            if (rank != 0) data = tmp;
            delete[] tmp;
        }
        else if (std::is_same<T, int>::value) MPI_Bcast(&data, 1, MPI_INT, 0, comm);
        else if (std::is_same<T, std::ptrdiff_t>::value) MPI_Bcast(&data, 1, MPI_AINT, 0, comm);
        else if (std::is_same<T, bool>::value) MPI_Bcast(&data, 1, MPI_CXX_BOOL, 0, comm);
        else if (std::is_same<T, float>::value) {
            MPI_Bcast(&data, 1, MPI_FLOAT, 0, comm);
            if (unit > 0) data /= unit;
        }
        else if (std::is_same<T, double>::value) {
            MPI_Bcast(&data, 1, MPI_DOUBLE, 0, comm);
            if (unit > 0) data /= unit;
        }
        else throw std::invalid_argument("readxml_bcast: input data type not implemented currently");
        
        if (rank == 0) std::cout << "Broadcasted " << path << std::endl;
        return true;
    }
    else
    {
        return false;
    }
}

#endif /* input_hpp */
