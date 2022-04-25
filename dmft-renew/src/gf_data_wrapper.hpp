//
//  gf_data_wrapper.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef gf_data_wrapper_hpp
#define gf_data_wrapper_hpp

#include <algorithm>
#include <complex>
#include <array>
#include <Eigen/Core>
#include <mpi.h>
#include <iostream>
#include <iomanip>

// Get underlying data type of std::size_t
#include <cstdint>
#include <climits>
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
   #error "What is happening here?"
#endif

/*****************************************************************************************************
Storage class template
*******************************************************************************************************/

// Primary template, fixed-size array and unit square matrix
template<typename _T, int _n0, int _n1, int _nm> class SqMatArrayStorage {
    static_assert(_n0 >= 0 && _n1 >= 0 && _nm >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, _nm, _n0 * _n1 * _nm> DataType;
    SqMatArrayStorage() {}
    SqMatArrayStorage(const std::size_t n0, const std::size_t n1, const std::size_t nm) {assert(n0 == _n0 && n1 == _n1 && nm == _nm);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    static constexpr std::size_t dim0(void) {return _n0;}
    static constexpr std::size_t dim1(void) {return _n1;}
    static constexpr std::size_t dimm(void) {return _nm;}
    static constexpr std::size_t size(void) {return _n0 * _n1;}
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {assert(n0 == _n0 && n1 == _n1 && nm == _nm);}
protected:
    DataType m_data;
};

// Below are partial specializations for the 7 dynamic cases
template<typename _T, int _n1, int _nm> class SqMatArrayStorage<_T, Eigen::Dynamic, _n1, _nm> {
    static_assert(_n1 >= 0 && _nm >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, _nm, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n0(0) {}
    SqMatArrayStorage(const std::size_t n0, const std::size_t n1, const std::size_t nm) : m_data(_nm, n0 * (_n1 * _nm)), m_n0(n0) {assert(n1 == _n1 && nm == _nm);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    std::size_t dim0(void) const {return m_n0;}
    static constexpr std::size_t dim1(void) {return _n1;}
    static constexpr std::size_t dimm(void) {return _nm;}
    std::size_t size(void) const {return m_n0 * _n1;}
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {assert(n1 == _n1 && nm == _nm); m_data.resize(Eigen::NoChange, n0 * (_n1 * _nm)); m_n0 = n0;}
protected:
    DataType m_data;
    std::size_t m_n0;   // Cannot use m_data.cols() to deduce m_n0 because _n1 or _nm could be zero
};

template<typename _T, int _n0, int _nm> class SqMatArrayStorage<_T, _n0, Eigen::Dynamic, _nm> {
    static_assert(_n0 >= 0 && _nm >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, _nm, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n1(0) {}
    SqMatArrayStorage(const std::size_t n0, const std::size_t n1, const std::size_t nm) : m_data(_nm, (_n0 * _nm) * n1), m_n1(n1) {assert(n0 == _n0 && nm == _nm);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    static constexpr std::size_t dim0(void) {return _n0;}
    std::size_t dim1(void) const {return m_n1;}
    static constexpr std::size_t dimm(void) {return _nm;}
    std::size_t size(void) const {return _n0 * m_n1;}
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {assert(n0 == _n0 && nm == _nm); m_data.resize(Eigen::NoChange, (_n0 * _nm) * n1); m_n1 = n1;}
protected:
    DataType m_data;
    std::size_t m_n1;
};

template<typename _T, int _nm> class SqMatArrayStorage<_T, Eigen::Dynamic, Eigen::Dynamic, _nm> {
    static_assert(_nm >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, _nm, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n0(0), m_n1(0) {}
    SqMatArrayStorage(const std::size_t n0, const std::size_t n1, const std::size_t nm) : m_data(_nm, n0 * n1 * _nm), m_n0(n0), m_n1(n1) {assert(nm == _nm);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    std::size_t dim0(void) const {return m_n0;}
    std::size_t dim1(void) const {return m_n1;}
    static constexpr std::size_t dimm(void) {return _nm;}
    std::size_t size(void) const {return m_n0 * m_n1;}
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {assert(nm == _nm); m_data.resize(Eigen::NoChange, n0 * n1 * _nm); m_n0 = n0; m_n1 = n1;}
protected:
    DataType m_data;
    std::size_t m_n0, m_n1;  // Cannot use m_data.cols() to deduce the other because m_n0 or m_n1 or m_data.rows() could be zero
};

template<typename _T, int _n0, int _n1> class SqMatArrayStorage<_T, _n0, _n1, Eigen::Dynamic> {
    static_assert(_n0 >= 0 && _n1 >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic> DataType;
    SqMatArrayStorage() {}
    SqMatArrayStorage(const std::size_t n0, const std::size_t n1, const std::size_t nm) : m_data(nm, (_n0 * _n1) * nm) {assert(n0 == _n0 && n1 == _n1);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    static constexpr std::size_t dim0(void) {return _n0;}
    static constexpr std::size_t dim1(void) {return _n1;}
    std::size_t dimm(void) const {return m_data.rows();}
    static constexpr std::size_t size(void) {return _n0 * _n1;}
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {assert(n0 == _n0 && n1 == _n1); m_data.resize(nm, (_n0 * _n1) * nm);}
protected:
    DataType m_data;
};

template<typename _T, int _n1> class SqMatArrayStorage<_T, Eigen::Dynamic, _n1, Eigen::Dynamic> {
    static_assert(_n1 >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n0(0) {}
    SqMatArrayStorage(const std::size_t n0, const std::size_t n1, const std::size_t nm) : m_data(nm, n0 * _n1 * nm), m_n0(n0) {assert(n1 == _n1);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    std::size_t dim0(void) const {return m_n0;}
    static constexpr std::size_t dim1(void) {return _n1;}
    std::size_t dimm(void) const {return m_data.rows();}
    std::size_t size(void) const {return m_n0 * _n1;}
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {assert(n1 == _n1); m_data.resize(nm, n0 * _n1 * nm); m_n0 = n0;}
protected:
    DataType m_data;
    std::size_t m_n0;
};

template<typename _T, int _n0> class SqMatArrayStorage<_T, _n0, Eigen::Dynamic, Eigen::Dynamic> {
    static_assert(_n0 >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n1(0) {}
    SqMatArrayStorage(const std::size_t n0, const std::size_t n1, const std::size_t nm) : m_data(nm, _n0 * n1 * nm), m_n1(n1) {assert(n0 == _n0);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    static constexpr std::size_t dim0(void) {return _n0;}
    std::size_t dim1(void) const {return m_n1;}
    std::size_t dimm(void) const {return m_data.rows();}
    std::size_t size(void) const {return _n0 * m_n1;}
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {assert(n0 == _n0); m_data.resize(nm, _n0 * n1 * nm); m_n1 = n1;}
protected:
    DataType m_data;
    std::size_t m_n1;
};

template<typename _T> class SqMatArrayStorage<_T, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> {
public:
    typedef Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n0(0), m_n1(0) {}
    SqMatArrayStorage(const std::size_t n0, const std::size_t n1, const std::size_t nm) : m_data(nm, n0 * n1 * nm), m_n0(n0), m_n1(n1) {}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    std::size_t dim0(void) const {return m_n0;}
    std::size_t dim1(void) const {return m_n1;}
    std::size_t dimm(void) const {return m_data.rows();}
    std::size_t size(void) const {return m_n0 * m_n1;}
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {m_data.resize(nm, n0 * n1 * nm); m_n0 = n0; m_n1 = n1;}
protected:
    DataType m_data;
    std::size_t m_n0, m_n1;  // Cannot use m_data.cols() to deduce the other because m_n0 or m_n1 or m_data.rows() could be zero
};

/*****************************************************************************************************
Expression class template for MPI partitioning
*******************************************************************************************************/

// Helper function for obtaining partition size and start
inline void mostEvenPart(const std::size_t size, const int psize, const int prank, std::size_t& partsize, std::size_t& partstart) {
    const int r0 = psize - static_cast<int>(size) % psize;
    const std::size_t bbsize = size / psize;
    if (prank < r0) {
        partsize = bbsize;
        partstart = prank * bbsize;
    }
    else {
        partsize = bbsize + 1;
        partstart = prank * (bbsize + 1) - r0;
    }
}

// Flat partition type
template<typename SqMatArrType>
class MastFlatPart {
public:
    explicit MastFlatPart(SqMatArrType& sqmatarr) : m_sqmatarr(sqmatarr) {
        //std::cout << "ctor" << std::endl;
    }
    MastFlatPart(const MastFlatPart&) = default;  // Default copy and move constructors are good for reference member
    MastFlatPart(MastFlatPart&&) = default;
    MastFlatPart& operator=(const MastFlatPart&) = delete;
    MastFlatPart& operator=(MastFlatPart&&) = delete;
    ~MastFlatPart() = default;
    
    std::size_t size() const {return m_sqmatarr.m_mastsize_flat;}
    std::size_t start() const {return m_sqmatarr.m_maststart_flat;}
    
    std::array<std::size_t, 2> global2dIndex(std::size_t i) const {
        assert(i < m_sqmatarr.m_mastsize_flat);
        i += m_sqmatarr.m_maststart_flat;
        return std::array<std::size_t, 2>{i / m_sqmatarr.dim1(), i % m_sqmatarr.dim1()};
    }
    
    typename SqMatArrType::ColsBlockXpr operator()() {
        return m_sqmatarr.m_data.middleCols(m_sqmatarr.m_maststart_flat * m_sqmatarr.dimm(), m_sqmatarr.m_mastsize_flat * m_sqmatarr.dimm());
    }
    typename SqMatArrType::ConstColsBlockXpr operator()() const {
        return m_sqmatarr.m_data.middleCols(m_sqmatarr.m_maststart_flat * m_sqmatarr.dimm(), m_sqmatarr.m_mastsize_flat * m_sqmatarr.dimm());
    }
    typename SqMatArrType::NColsBlockXpr operator[](const std::size_t i) {
        assert(i < m_sqmatarr.m_mastsize_flat);
        //std::cout << "non-const" << std::endl;
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::UnitSizeAtCompileTime>((i + m_sqmatarr.m_maststart_flat) * m_sqmatarr.dimm(), m_sqmatarr.dimm());
    }
    typename SqMatArrType::ConstNColsBlockXpr operator[](const std::size_t i) const {
        assert(i < m_sqmatarr.m_mastsize_flat);
        //std::cout << "const" << std::endl;
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::UnitSizeAtCompileTime>((i + m_sqmatarr.m_maststart_flat) * m_sqmatarr.dimm(), m_sqmatarr.dimm());
    }
    typename SqMatArrType::Scalar& operator()(const std::size_t i, const std::size_t im0, const std::size_t im1) {
        assert(i < m_sqmatarr.m_mastsize_flat && im0 < m_sqmatarr.dimm() && im1 < m_sqmatarr.dimm());
        return m_sqmatarr.m_data(im0, (i + m_sqmatarr.m_maststart_flat) * m_sqmatarr.dimm() + im1);
    }
    typename SqMatArrType::Scalar operator()(const std::size_t i, const std::size_t im0, const std::size_t im1) const {
        assert(i < m_sqmatarr.m_mastsize_flat && im0 < m_sqmatarr.dimm() && im1 < m_sqmatarr.dimm());
        return m_sqmatarr.m_data(im0, (i + m_sqmatarr.m_maststart_flat) * m_sqmatarr.dimm() + im1);
    }
    
    // Sum-reduce _data on all local processes and the results are separately held in the mastered partition on each process
    void sum2mastPart();
    // Gather each piece governed by each local process and broadcast the result to all processes.
    // Note every process holds the full-sized _data, but it is assumed that each process governs
    // a piece of _data. There is no need to really scatter _data to processes and reallocate
    // _data every time when entering DMFT equations from QMC run, because _data upon exiting QMC
    // run on every process is intact.
    void allGather();
    
private:
    SqMatArrType& m_sqmatarr;
};

template<typename SqMatArrType>
void MastFlatPart<SqMatArrType>::sum2mastPart() {
    static_assert(std::is_same<typename SqMatArrType::Scalar, double>::value || std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value);
    if (m_sqmatarr.m_is_inter) throw std::invalid_argument("MPI communicator is an intercommunicator prohibiting in-place Reduce!");
    const std::size_t nmsq = m_sqmatarr.dimm() * m_sqmatarr.dimm();
    std::size_t chunksize, chunkstart;
    for (int dest = 0; dest < m_sqmatarr.m_psize; ++dest) {
        // Get size and start of the destination process
        if (m_sqmatarr.m_prank == dest) {
            chunksize = m_sqmatarr.m_mastsize_flat * nmsq;
            chunkstart = m_sqmatarr.m_maststart_flat * nmsq;
        }
        // Broadcast size and start of the destination process to all processes
        MPI_Bcast(&chunksize, 1, my_MPI_SIZE_T, dest, m_sqmatarr.m_comm);
        MPI_Bcast(&chunkstart, 1, my_MPI_SIZE_T, dest, m_sqmatarr.m_comm);
        if (m_sqmatarr.m_prank == dest) {
            if constexpr (std::is_same<typename SqMatArrType::Scalar, double>::value)
                MPI_Reduce(MPI_IN_PLACE, m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE, MPI_SUM, dest, m_sqmatarr.m_comm);
            else if (std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value)
                MPI_Reduce(MPI_IN_PLACE, m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE_COMPLEX, MPI_SUM, dest, m_sqmatarr.m_comm);
        }
        else {
            if constexpr (std::is_same<typename SqMatArrType::Scalar, double>::value)
                MPI_Reduce(m_sqmatarr.m_data.data() + chunkstart, m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE, MPI_SUM, dest, m_sqmatarr.m_comm);
            else if (std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value)
                MPI_Reduce(m_sqmatarr.m_data.data() + chunkstart, m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE_COMPLEX, MPI_SUM, dest, m_sqmatarr.m_comm);
        }
    }
}

template<typename SqMatArrType>
void MastFlatPart<SqMatArrType>::allGather() {
    static_assert(std::is_same<typename SqMatArrType::Scalar, double>::value || std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value);
    const std::size_t nmsq = m_sqmatarr.dimm() * m_sqmatarr.dimm();
    std::size_t chunksize, chunkstart;
    for (int src = 0; src < m_sqmatarr.m_psize; ++src) {
        // Get size and start of the source process
        if (m_sqmatarr.m_prank == src) {
            chunksize = m_sqmatarr.m_mastsize_flat * nmsq;
            chunkstart = m_sqmatarr.m_maststart_flat * nmsq;
        }
        // Broadcast size and start of the source process to all processes
        MPI_Bcast(&chunksize, 1, my_MPI_SIZE_T, src, m_sqmatarr.m_comm);
        MPI_Bcast(&chunkstart, 1, my_MPI_SIZE_T, src, m_sqmatarr.m_comm);
        if constexpr (std::is_same<typename SqMatArrType::Scalar, double>::value)
            MPI_Bcast(m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE, src, m_sqmatarr.m_comm);
        else if (std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value)
            MPI_Bcast(m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE_COMPLEX, src, m_sqmatarr.m_comm);
    }
}

// Type for partitioning only the first dimension
template<typename SqMatArrType>
class MastDim0Part {
public:
    explicit MastDim0Part(SqMatArrType& sqmatarr) : m_sqmatarr(sqmatarr) {}
    MastDim0Part(const MastDim0Part&) = default;  // Default copy and move constructors are good for reference member
    MastDim0Part(MastDim0Part&&) = default;
    MastDim0Part& operator=(const MastDim0Part&) = delete;
    MastDim0Part& operator=(MastDim0Part&&) = delete;
    ~MastDim0Part() = default;
    
    std::size_t dim0() const {return m_sqmatarr.m_mastsize_dim0;}
    std::size_t start() const {return m_sqmatarr.m_maststart_dim0;}
    
    typename SqMatArrType::ColsBlockXpr operator()() {
        return m_sqmatarr.m_data.middleCols(m_sqmatarr.m_maststart_dim0 * m_sqmatarr.dim1() * m_sqmatarr.dimm(), m_sqmatarr.m_mastsize_dim0 * m_sqmatarr.dim1() * m_sqmatarr.dimm());
    }
    typename SqMatArrType::ConstColsBlockXpr operator()() const {
        return m_sqmatarr.m_data.middleCols(m_sqmatarr.m_maststart_dim0 * m_sqmatarr.dim1() * m_sqmatarr.dimm(), m_sqmatarr.m_mastsize_dim0 * m_sqmatarr.dim1() * m_sqmatarr.dimm());
    }
    typename SqMatArrType::NColsBlockXpr operator()(const std::size_t i0, const std::size_t i1) {
        assert(i0 < m_sqmatarr.m_mastsize_dim0 && i1 < m_sqmatarr.dim1());
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::UnitSizeAtCompileTime>(((i0 + m_sqmatarr.m_maststart_dim0) * m_sqmatarr.dim1() + i1) * m_sqmatarr.dimm(), m_sqmatarr.dimm());
    }
    typename SqMatArrType::ConstNColsBlockXpr operator()(const std::size_t i0, const std::size_t i1) const {
        assert(i0 < m_sqmatarr.m_mastsize_dim0 && i1 < m_sqmatarr.dim1());
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::UnitSizeAtCompileTime>(((i0 + m_sqmatarr.m_maststart_dim0) * m_sqmatarr.dim1() + i1) * m_sqmatarr.dimm(), m_sqmatarr.dimm());
    }
    typename SqMatArrType::Scalar& operator()(const std::size_t i0, const std::size_t i1, const std::size_t im0, const std::size_t im1) {
        assert(i0 < m_sqmatarr.m_mastsize_dim0 && i1 < m_sqmatarr.dim1() && im0 < m_sqmatarr.dimm() && im1 < m_sqmatarr.dimm());
        return m_sqmatarr.m_data(im0, ((i0 + m_sqmatarr.m_maststart_dim0) * m_sqmatarr.dim1() + i1) * m_sqmatarr.dimm() + im1);
    }
    typename SqMatArrType::Scalar operator()(const std::size_t i0, const std::size_t i1, const std::size_t im0, const std::size_t im1) const {
        assert(i0 < m_sqmatarr.m_mastsize_dim0 && i1 < m_sqmatarr.dim1() && im0 < m_sqmatarr.dimm() && im1 < m_sqmatarr.dimm());
        return m_sqmatarr.m_data(im0, ((i0 + m_sqmatarr.m_maststart_dim0) * m_sqmatarr.dim1() + i1) * m_sqmatarr.dimm() + im1);
    }
    
    // Sum-reduce _data on all local processes and the results are separately held in the mastered partition on each process
    void sum2mastPart();
    // Gather each piece governed by each local process and broadcast the result to all processes.
    // Note every process holds the full-sized _data, but it is assumed that each process governs
    // a piece of _data. There is no need to really scatter _data to processes and reallocate
    // _data every time when entering DMFT equations from QMC run, because _data upon exiting QMC
    // run on every process is intact.
    void allGather();
    
private:
    SqMatArrType& m_sqmatarr;
};

template<typename SqMatArrType>
void MastDim0Part<SqMatArrType>::sum2mastPart() {
    static_assert(std::is_same<typename SqMatArrType::Scalar, double>::value || std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value);
    if (m_sqmatarr.m_is_inter) throw std::invalid_argument("MPI communicator is an intercommunicator prohibiting in-place Reduce!");
    const std::size_t dim1nmsq = m_sqmatarr.dim1() * m_sqmatarr.dimm() * m_sqmatarr.dimm();
    std::size_t chunksize, chunkstart;
    for (int dest = 0; dest < m_sqmatarr.m_psize; ++dest) {
        // Get size and start of the destination process
        if (m_sqmatarr.m_prank == dest) {
            chunksize = m_sqmatarr.m_mastsize_dim0 * dim1nmsq;
            chunkstart = m_sqmatarr.m_maststart_dim0 * dim1nmsq;
        }
        // Broadcast size and start of the destination process to all processes
        MPI_Bcast(&chunksize, 1, my_MPI_SIZE_T, dest, m_sqmatarr.m_comm);
        MPI_Bcast(&chunkstart, 1, my_MPI_SIZE_T, dest, m_sqmatarr.m_comm);
        if (m_sqmatarr.m_prank == dest) {
            if constexpr (std::is_same<typename SqMatArrType::Scalar, double>::value)
                MPI_Reduce(MPI_IN_PLACE, m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE, MPI_SUM, dest, m_sqmatarr.m_comm);
            else if (std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value)
                MPI_Reduce(MPI_IN_PLACE, m_sqmatarr.m_data.data() + chunkstart, chunksize, MPI_DOUBLE_COMPLEX, MPI_SUM, dest, m_sqmatarr.m_comm);
        }
        else {
            if constexpr (std::is_same<typename SqMatArrType::Scalar, double>::value)
                MPI_Reduce(m_sqmatarr.m_data.data() + chunkstart, m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE, MPI_SUM, dest, m_sqmatarr.m_comm);
            else if (std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value)
                MPI_Reduce(m_sqmatarr.m_data.data() + chunkstart, m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE_COMPLEX, MPI_SUM, dest, m_sqmatarr.m_comm);
        }
    }
}

template<typename SqMatArrType>
void MastDim0Part<SqMatArrType>::allGather() {
    static_assert(std::is_same<typename SqMatArrType::Scalar, double>::value || std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value);
    const std::size_t dim1nmsq = m_sqmatarr.dim1() * m_sqmatarr.dimm() * m_sqmatarr.dimm();
    std::size_t chunksize, chunkstart;
    for (int src = 0; src < m_sqmatarr.m_psize; ++src) {
        // Get size and start of the source process
        if (m_sqmatarr.m_prank == src) {
            chunksize = m_sqmatarr.m_mastsize_dim0 * dim1nmsq;
            chunkstart = m_sqmatarr.m_maststart_dim0 * dim1nmsq;
        }
        // Broadcast size and start of the source process to all processes
        MPI_Bcast(&chunksize, 1, my_MPI_SIZE_T, src, m_sqmatarr.m_comm);
        MPI_Bcast(&chunkstart, 1, my_MPI_SIZE_T, src, m_sqmatarr.m_comm);
        if constexpr (std::is_same<typename SqMatArrType::Scalar, double>::value)
            MPI_Bcast(m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE, src, m_sqmatarr.m_comm);
        else if (std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value)
            MPI_Bcast(m_sqmatarr.m_data.data() + chunkstart, static_cast<int>(chunksize), MPI_DOUBLE_COMPLEX, src, m_sqmatarr.m_comm);
    }
}

/*****************************************************************************************************
Formal data wrapper class template
*******************************************************************************************************/

template<typename _Scalar, int _n0, int _n1, int _nm>
class SqMatArray : public SqMatArrayStorage<_Scalar, _n0, _n1, _nm> {
    template<typename SqMatArrType>
    friend class MastFlatPart;
    template<typename SqMatArrType>
    friend class MastDim0Part;

    void mpiImagPart() {
        mostEvenPart(this->size(), m_psize, m_prank, m_mastsize_flat, m_maststart_flat);  // Flat partition
        mostEvenPart(this->dim0(), m_psize, m_prank, m_mastsize_dim0, m_maststart_dim0);  // Partition only the first dimension
    }
    
public:
    typedef _Scalar Scalar;
    static constexpr int UnitSizeAtCompileTime = _nm;
    typedef typename Eigen::DenseBase<typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType>::ColsBlockXpr ColsBlockXpr;
    typedef typename Eigen::DenseBase<typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType>::ConstColsBlockXpr ConstColsBlockXpr;
    // Below templates accept _nm being fixed size or dynamic, and in the former case the runtime size must equal _nm
    typedef typename Eigen::DenseBase<typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType>::template NColsBlockXpr<_nm>::Type NColsBlockXpr;
    typedef typename Eigen::DenseBase<typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType>::template ConstNColsBlockXpr<_nm>::Type ConstNColsBlockXpr;
    
    explicit SqMatArray(const MPI_Comm& comm = MPI_COMM_SELF) : m_comm(comm) {
        MPI_Comm_size(comm, &m_psize); MPI_Comm_rank(comm, &m_prank); MPI_Comm_test_inter(comm, &m_is_inter);
        mpiImagPart();
    }
    SqMatArray(const std::size_t n0, const std::size_t n1, const std::size_t nm, const MPI_Comm& comm = MPI_COMM_SELF) : SqMatArrayStorage<_Scalar, _n0, _n1, _nm>(n0, n1, nm), m_comm(comm) {
        MPI_Comm_size(comm, &m_psize); MPI_Comm_rank(comm, &m_prank); MPI_Comm_test_inter(comm, &m_is_inter);
        mpiImagPart();
    }
    SqMatArray(const SqMatArray&) = default;
    SqMatArray(SqMatArray&&) = default;
    SqMatArray& operator=(const SqMatArray&) = default;
    SqMatArray& operator=(SqMatArray&&) = default;
    ~SqMatArray() = default;
    
    const MPI_Comm& mpiCommunicator(void) const {return m_comm;}
    void mpiCommunicator(const MPI_Comm& comm) {
        m_comm = comm;
        MPI_Comm_size(comm, &m_psize); MPI_Comm_rank(comm, &m_prank); MPI_Comm_test_inter(comm, &m_is_inter);
        mpiImagPart();
    }
    int processSize() const {return m_psize;}
    int processRank() const {return m_prank;}
    
    // Expose the underlying Eigen::Matrix object for global manipulation (should not do resize)
    typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType& operator()() {return this->m_data;}
    const typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType& operator()() const {return this->m_data;}
    
    // Just sequentially access each unit matrix, which becomes intuitive when _n0 = 1 or _n1 = 1.
    NColsBlockXpr operator[](const std::size_t i) {
        assert(i < this->size());
        return this->m_data.template middleCols<_nm>(i * this->dimm(), this->dimm());
    }
    ConstNColsBlockXpr operator[](const std::size_t i) const {
        assert(i < this->size());
        return this->m_data.template middleCols<_nm>(i * this->dimm(), this->dimm());
    }
    
    // Overload operator () to intuitively access each unit matrix
    NColsBlockXpr operator()(const std::size_t i0, const std::size_t i1) {  // Non-const version
        assert(i0 < this->dim0() && i1 < this->dim1());
        return this->m_data.template middleCols<_nm>((i0 * this->dim1() + i1) * this->dimm(), this->dimm());
    }
    ConstNColsBlockXpr operator()(const std::size_t i0, const std::size_t i1) const {   // Const version
        assert(i0 < this->dim0() && i1 < this->dim1());
        return this->m_data.template middleCols<_nm>((i0 * this->dim1() + i1) * this->dimm(), this->dimm());
    }
    
    // Directly access element
    _Scalar& operator()(const std::size_t i0, const std::size_t i1, const std::size_t im0, const std::size_t im1) {
        assert(i0 < this->dim0() && i1 < this->dim1() && im0 < this->dimm() && im1 < this->dimm());
        return this->m_data(im0, (i0 * this->dim1() + i1) * this->dimm() + im1);
    }
    _Scalar operator()(const std::size_t i0, const std::size_t i1, const std::size_t im0, const std::size_t im1) const {
        assert(i0 < this->dim0() && i1 < this->dim1() && im0 < this->dimm() && im1 < this->dimm());
        return this->m_data(im0, (i0 * this->dim1() + i1) * this->dimm() + im1);
    }
    
    // Get MPI partition. Note: if a variable is defined to store this returned object, its type's const-ness
    // must exactly follow those of these return types, e.g., for a const SqMatArray<...>, the variable's declaration
    // must be const MastFlatPart<const SqMatArray<...> >, or a compilation error would occur to complain that calls to
    // some of the variable's methods are incompatible with their return type in terms of the const-ness. If auto is used
    // for the variable's type, the second const-ness is retained, but the first const-ness is not, so remember to prepend
    // const if needed.
    MastFlatPart<SqMatArray> mastFlatPart() {return MastFlatPart<SqMatArray>(*this);}
    const MastFlatPart<const SqMatArray> mastFlatPart() const {return MastFlatPart<const SqMatArray>(*this);}
    MastDim0Part<SqMatArray> mastDim0Part() {return MastDim0Part<SqMatArray>(*this);}
    const MastDim0Part<const SqMatArray> mastDim0Part() const {return MastDim0Part<const SqMatArray>(*this);}
    
    // Override SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::resize(n0, n1, nm) to include re-partitioning.
    // The base resize method doesn't need to be virtual because one doesn't want to refer to this class by the base storage class
    void resize(const std::size_t n0, const std::size_t n1, const std::size_t nm) {
        const std::size_t oldsize = this->size();
        SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::resize(n0, n1, nm);
        if (this->size() != oldsize) mpiImagPart();
    }
    
    void allSum();   // allSum-reduce the whole m_data on all local processes
    void bCast(const int rank);   // Broadcast the whole m_data on process "rank" to other processes
    
protected:
    MPI_Comm m_comm;
    int m_psize, m_prank, m_is_inter;
    std::size_t m_mastsize_flat, m_maststart_flat, m_mastsize_dim0, m_maststart_dim0;
};

template<typename _Scalar, int _n0, int _n1, int _nm>
void SqMatArray<_Scalar, _n0, _n1, _nm>::allSum() {
    static_assert(std::is_same<_Scalar, double>::value || std::is_same<_Scalar, std::complex<double> >::value);
    if (m_is_inter) throw std::invalid_argument("MPI communicator is an intercommunicator prohibiting in-place Allreduce!");
    if constexpr (std::is_same<_Scalar, double>::value)
        MPI_Allreduce(MPI_IN_PLACE, this->m_data.data(), static_cast<int>(this->m_data.size()), MPI_DOUBLE, MPI_SUM, m_comm);
    else if (std::is_same<_Scalar, std::complex<double> >::value)
        MPI_Allreduce(MPI_IN_PLACE, this->m_data.data(), static_cast<int>(this->m_data.size()), MPI_DOUBLE_COMPLEX, MPI_SUM, m_comm);
}

template<typename _Scalar, int _n0, int _n1, int _nm>
void SqMatArray<_Scalar, _n0, _n1, _nm>::bCast(const int rank) {
    static_assert(std::is_same<_Scalar, double>::value || std::is_same<_Scalar, std::complex<double> >::value);
    if constexpr (std::is_same<_Scalar, double>::value)
        MPI_Bcast(this->m_data.data(), static_cast<int>(this->m_data.size()), MPI_DOUBLE, rank, m_comm);
    else if (std::is_same<_Scalar, std::complex<double> >::value)
        MPI_Bcast(this->m_data.data(), static_cast<int>(this->m_data.size()), MPI_DOUBLE_COMPLEX, rank, m_comm);
}








// Overload the insertion operator
template<typename Scalar, int n0, int n1, int nm>
std::ostream& operator<<(std::ostream& os, const SqMatArray<Scalar, n0, n1, nm>& sqmats) {
//    int charspernum;
//    std::array<std::size_t, 2> i2d;
//    MPI_Offset disp, offset;
//    MPI_Datatype NumAsStr;
//
//    if constexpr (std::is_same<ScalarT, std::complex<double> >::value) charspernum = 30;
//    else charspernum = 15;
//    i2d = vecsqmat.index2DinPart(0);
//    disp = static_cast<MPI_Offset>((i2d[0] + i2d[1] * vecsqmat.dim0()));
//
//
//    MPI_Type_contiguous(charspernum, MPI_CHAR, &NumAsStr);
//    MPI_Type_commit(&NumAsStr);
    const int w = static_cast<int>(os.precision()) + 3;
    std::size_t i0, i1, im0, im1;
    for (i1 = 0; i1 < sqmats.dim1(); ++i1) {
        for (i0 = 0; i0 < sqmats.dim0(); ++i0) {
            for (im1 = 0; im1 < sqmats.dimm(); ++im1) {
                for (im0 = 0; im0 < sqmats.dimm(); ++im0) {
                    if constexpr (std::is_same<Scalar, std::complex<double> >::value) os << std::setw(w) << sqmats(i0, i1, im0, im1).real() << " "
                        << std::setw(w) << sqmats(i0, i1, im0, im1).imag() << "  ";
                    else os << std::setw(w) << sqmats(i0, i1, im0, im1) << "  ";
                }
            }
        }
        os << std::endl;
    }
    return os;
}

// Overload the extraction operator
template<typename Scalar, int n0, int n1, int nm>
std::istream& operator>>(std::istream& is, SqMatArray<Scalar, n0, n1, nm>& sqmats) {
    std::size_t i0, i1, im0, im1;
    double real, imag;
    for (i1 = 0; i1 < sqmats.dim1(); ++i1) {
        for (i0 = 0; i0 < sqmats.dim0(); ++i0) {
            for (im1 = 0; im1 < sqmats.dimm(); ++im1) {
                for (im0 = 0; im0 < sqmats.dimm(); ++im0) {
                    if constexpr (std::is_same<Scalar, std::complex<double> >::value) {
                        is >> real >> imag;
                        sqmats(i0, i1, im0, im1).real(real);
                        sqmats(i0, i1, im0, im1).imag(imag);
                    }
                    else is >> sqmats(i0, i1, im0, im1);
                }
            }
        }
    }
    return is;
}






typedef SqMatArray<double, 2, 1, Eigen::Dynamic> SqMatArray21Xd;
typedef SqMatArray<std::complex<double>, 2, 1, Eigen::Dynamic> SqMatArray21Xcd;

typedef SqMatArray<double, 2, Eigen::Dynamic, Eigen::Dynamic> SqMatArray2XXd;
typedef SqMatArray<std::complex<double>, 2, Eigen::Dynamic, Eigen::Dynamic> SqMatArray2XXcd;


#endif /* gf_data_wrapper_hpp */
