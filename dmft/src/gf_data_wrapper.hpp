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
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <charconv>
#include <iterator>

/*****************************************************************************************************
Storage class template
*******************************************************************************************************/

// Primary template, fixed-size array and unit square matrix
template<typename _T, int _n0, int _n1, int _nm> class SqMatArrayStorage {
    static_assert(_n0 >= 0 && _n1 >= 0 && _nm >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, _nm, _n0 * _n1 * _nm> DataType;
    SqMatArrayStorage() {}
    SqMatArrayStorage(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {assert(n0 == _n0 && n1 == _n1 && nm == _nm);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    static constexpr Eigen::Index dim0(void) {return _n0;}
    static constexpr Eigen::Index dim1(void) {return _n1;}
    static constexpr Eigen::Index dimm(void) {return _nm;}
    static constexpr Eigen::Index size(void) {return _n0 * _n1;}
    void resize(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {assert(n0 == _n0 && n1 == _n1 && nm == _nm);}
protected:
    DataType m_data;
};

// Below are partial specializations for the 7 dynamic cases
template<typename _T, int _n1, int _nm> class SqMatArrayStorage<_T, Eigen::Dynamic, _n1, _nm> {
    static_assert(_n1 >= 0 && _nm >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, _nm, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n0(0) {}
    SqMatArrayStorage(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) : m_data(_nm, n0 * (_n1 * _nm)), m_n0(n0) {assert(n1 == _n1 && nm == _nm);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    Eigen::Index dim0(void) const {return m_n0;}
    static constexpr Eigen::Index dim1(void) {return _n1;}
    static constexpr Eigen::Index dimm(void) {return _nm;}
    Eigen::Index size(void) const {return m_n0 * _n1;}
    void resize(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {assert(n1 == _n1 && nm == _nm); m_data.resize(Eigen::NoChange, n0 * (_n1 * _nm)); m_n0 = n0;}
protected:
    DataType m_data;
    Eigen::Index m_n0;   // Cannot use m_data.cols() to deduce m_n0 because _n1 or _nm could be zero
};

template<typename _T, int _n0, int _nm> class SqMatArrayStorage<_T, _n0, Eigen::Dynamic, _nm> {
    static_assert(_n0 >= 0 && _nm >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, _nm, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n1(0) {}
    SqMatArrayStorage(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) : m_data(_nm, (_n0 * _nm) * n1), m_n1(n1) {assert(n0 == _n0 && nm == _nm);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    static constexpr Eigen::Index dim0(void) {return _n0;}
    Eigen::Index dim1(void) const {return m_n1;}
    static constexpr Eigen::Index dimm(void) {return _nm;}
    Eigen::Index size(void) const {return _n0 * m_n1;}
    void resize(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {assert(n0 == _n0 && nm == _nm); m_data.resize(Eigen::NoChange, (_n0 * _nm) * n1); m_n1 = n1;}
protected:
    DataType m_data;
    Eigen::Index m_n1;
};

template<typename _T, int _nm> class SqMatArrayStorage<_T, Eigen::Dynamic, Eigen::Dynamic, _nm> {
    static_assert(_nm >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, _nm, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n0(0), m_n1(0) {}
    SqMatArrayStorage(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) : m_data(_nm, n0 * n1 * _nm), m_n0(n0), m_n1(n1) {assert(nm == _nm);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    Eigen::Index dim0(void) const {return m_n0;}
    Eigen::Index dim1(void) const {return m_n1;}
    static constexpr Eigen::Index dimm(void) {return _nm;}
    Eigen::Index size(void) const {return m_n0 * m_n1;}
    void resize(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {assert(nm == _nm); m_data.resize(Eigen::NoChange, n0 * n1 * _nm); m_n0 = n0; m_n1 = n1;}
protected:
    DataType m_data;
    Eigen::Index m_n0, m_n1;  // Cannot use m_data.cols() to deduce the other because m_n0 or m_n1 or m_data.rows() could be zero
};

template<typename _T, int _n0, int _n1> class SqMatArrayStorage<_T, _n0, _n1, Eigen::Dynamic> {
    static_assert(_n0 >= 0 && _n1 >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic> DataType;
    SqMatArrayStorage() {}
    SqMatArrayStorage(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) : m_data(nm, (_n0 * _n1) * nm) {assert(n0 == _n0 && n1 == _n1);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    static constexpr Eigen::Index dim0(void) {return _n0;}
    static constexpr Eigen::Index dim1(void) {return _n1;}
    Eigen::Index dimm(void) const {return m_data.rows();}
    static constexpr Eigen::Index size(void) {return _n0 * _n1;}
    void resize(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {assert(n0 == _n0 && n1 == _n1); m_data.resize(nm, (_n0 * _n1) * nm);}
protected:
    DataType m_data;
};

template<typename _T, int _n1> class SqMatArrayStorage<_T, Eigen::Dynamic, _n1, Eigen::Dynamic> {
    static_assert(_n1 >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n0(0) {}
    SqMatArrayStorage(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) : m_data(nm, n0 * _n1 * nm), m_n0(n0) {assert(n1 == _n1);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    Eigen::Index dim0(void) const {return m_n0;}
    static constexpr Eigen::Index dim1(void) {return _n1;}
    Eigen::Index dimm(void) const {return m_data.rows();}
    Eigen::Index size(void) const {return m_n0 * _n1;}
    void resize(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {assert(n1 == _n1); m_data.resize(nm, n0 * _n1 * nm); m_n0 = n0;}
protected:
    DataType m_data;
    Eigen::Index m_n0;
};

template<typename _T, int _n0> class SqMatArrayStorage<_T, _n0, Eigen::Dynamic, Eigen::Dynamic> {
    static_assert(_n0 >= 0, "Fixed-size dimension of SqMatArrayStorage must be non-negative!");
public:
    typedef Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n1(0) {}
    SqMatArrayStorage(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) : m_data(nm, _n0 * n1 * nm), m_n1(n1) {assert(n0 == _n0);}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    static constexpr Eigen::Index dim0(void) {return _n0;}
    Eigen::Index dim1(void) const {return m_n1;}
    Eigen::Index dimm(void) const {return m_data.rows();}
    Eigen::Index size(void) const {return _n0 * m_n1;}
    void resize(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {assert(n0 == _n0); m_data.resize(nm, _n0 * n1 * nm); m_n1 = n1;}
protected:
    DataType m_data;
    Eigen::Index m_n1;
};

template<typename _T> class SqMatArrayStorage<_T, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> {
public:
    typedef Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic> DataType;
    SqMatArrayStorage() : m_n0(0), m_n1(0) {}
    SqMatArrayStorage(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) : m_data(nm, n0 * n1 * nm), m_n0(n0), m_n1(n1) {}
    SqMatArrayStorage(const SqMatArrayStorage&) = default;
    SqMatArrayStorage(SqMatArrayStorage&&) = default;
    SqMatArrayStorage& operator=(const SqMatArrayStorage&) = default;
    SqMatArrayStorage& operator=(SqMatArrayStorage&&) = default;
    virtual ~SqMatArrayStorage() {}
    Eigen::Index dim0(void) const {return m_n0;}
    Eigen::Index dim1(void) const {return m_n1;}
    Eigen::Index dimm(void) const {return m_data.rows();}
    Eigen::Index size(void) const {return m_n0 * m_n1;}
    void resize(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm) {m_data.resize(nm, n0 * n1 * nm); m_n0 = n0; m_n1 = n1;}
protected:
    DataType m_data;
    Eigen::Index m_n0, m_n1;  // Cannot use m_data.cols() to deduce the other because m_n0 or m_n1 or m_data.rows() could be zero
};

/*****************************************************************************************************
Expression class template for MPI partitioning
*******************************************************************************************************/

// Helper function for obtaining partition size and start
inline void mostEvenPart(const Eigen::Index size, const int psize, const int prank, Eigen::Index& partsize, Eigen::Index& partstart) {
    // Distribute residual from backward
    //const int r0 = psize - static_cast<int>(size) % psize;
    //const Eigen::Index bbsize = size / psize;
    //if (prank < r0) {
    //    partsize = bbsize;
    //    partstart = prank * bbsize;
    //}
    //else {
    //    partsize = bbsize + 1;
    //    partstart = prank * (bbsize + 1) - r0;
    //}
    // Distribute residual from forward
    const Eigen::Index r0 = size % psize;
    const Eigen::Index bbsize = size / psize;
    if (prank < r0) {
        partsize = bbsize + 1;
        partstart = prank * (bbsize + 1);
    }
    else {
        partsize = bbsize;
        partstart = prank * bbsize + r0;
    }
}

// Flat partition type
template<typename SqMatArrType>
class MastFlatPart {
public:
    explicit MastFlatPart(SqMatArrType& sqmatarr) : m_sqmatarr(sqmatarr) {
        mostEvenPart(sqmatarr.size(), sqmatarr.procSize(), sqmatarr.procRank(), m_size, m_displ);  // Flat partition
    }
    MastFlatPart(const MastFlatPart&) = default;  // Default copy and move constructors are good for reference member
    MastFlatPart(MastFlatPart&&) = default;
    MastFlatPart& operator=(const MastFlatPart&) = delete;
    MastFlatPart& operator=(MastFlatPart&&) = delete;
    ~MastFlatPart() = default;
    
    Eigen::Index size() const {return m_size;}
    Eigen::Index displ() const {return m_displ;}
    Eigen::Index dimm() const {return m_sqmatarr.dimm();}
    
    std::array<Eigen::Index, 2> global2dIndex(Eigen::Index i) const {
        assert(i < m_size);
        i += m_displ;
        return std::array<Eigen::Index, 2>{i / m_sqmatarr.dim1(), i % m_sqmatarr.dim1()};
    }
    
    typename SqMatArrType::ColsBlockXpr operator()() {
        return m_sqmatarr.m_data.middleCols(m_displ * m_sqmatarr.dimm(), m_size * m_sqmatarr.dimm());
    }
    typename SqMatArrType::ConstColsBlockXpr operator()() const {
        return m_sqmatarr.m_data.middleCols(m_displ * m_sqmatarr.dimm(), m_size * m_sqmatarr.dimm());
    }
    typename SqMatArrType::template NColsBlockXpr<SqMatArrType::DimmAtCompileTime> operator[](const Eigen::Index i) {
        assert(i < m_size);
        //std::cout << "non-const" << std::endl;
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::DimmAtCompileTime>((i + m_displ) * m_sqmatarr.dimm(), m_sqmatarr.dimm());
    }
    typename SqMatArrType::template ConstNColsBlockXpr<SqMatArrType::DimmAtCompileTime> operator[](const Eigen::Index i) const {
        assert(i < m_size);
        //std::cout << "const" << std::endl;
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::DimmAtCompileTime>((i + m_displ) * m_sqmatarr.dimm(), m_sqmatarr.dimm());
    }
    typename SqMatArrType::Scalar& operator()(const Eigen::Index i, const Eigen::Index im0, const Eigen::Index im1) {
        assert(i < m_size && im0 < m_sqmatarr.dimm() && im1 < m_sqmatarr.dimm());
        return m_sqmatarr.m_data(im0, (i + m_displ) * m_sqmatarr.dimm() + im1);
    }
    typename SqMatArrType::Scalar operator()(const Eigen::Index i, const Eigen::Index im0, const Eigen::Index im1) const {
        assert(i < m_size && im0 < m_sqmatarr.dimm() && im1 < m_sqmatarr.dimm());
        return m_sqmatarr.m_data(im0, (i + m_displ) * m_sqmatarr.dimm() + im1);
    }
    
    // Sum-reduce _data on all local processes and the results are separately held in the mastered partition on each process
    void sum2mastPart() {
        const Eigen::Index nmsq = m_sqmatarr.dimm() * m_sqmatarr.dimm();
        m_sqmatarr.sum2mastPart(m_size * nmsq, m_displ * nmsq);
    }
    // Gather each piece governed by each local process and broadcast the result to all processes.
    // Note every process holds the full-sized _data, but it is assumed that each process governs
    // a piece of _data. There is no need to really scatter _data to processes and reallocate
    // _data every time when entering DMFT equations from QMC run, because _data upon exiting QMC
    // run on every process is intact.
    void allGather() {
        const Eigen::Index nmsq = m_sqmatarr.dimm() * m_sqmatarr.dimm();
        m_sqmatarr.allGather(m_size * nmsq, m_displ * nmsq);
    }
    
private:
    SqMatArrType& m_sqmatarr;
    Eigen::Index m_size, m_displ;
};

/*
template<typename SqMatArrType>
void MastFlatPart<SqMatArrType>::sum2mastPart() {
    static_assert(std::is_same<typename SqMatArrType::Scalar, double>::value || std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value);
    if (m_sqmatarr.m_is_inter) throw std::invalid_argument("MPI communicator is an intercommunicator prohibiting in-place Reduce!");
    const Eigen::Index nmsq = m_sqmatarr.dimm() * m_sqmatarr.dimm();
    Eigen::Index chunksize, chunkstart;
    for (int dest = 0; dest < m_sqmatarr.m_psize; ++dest) {
        // Get size and start of the destination process
        if (m_sqmatarr.m_prank == dest) {
            chunksize = m_sqmatarr.m_mastsize_flat * nmsq;
            chunkstart = m_sqmatarr.m_mastdispl_flat * nmsq;
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
    const Eigen::Index nmsq = m_sqmatarr.dimm() * m_sqmatarr.dimm();
    Eigen::Index chunksize, chunkstart;
    for (int src = 0; src < m_sqmatarr.m_psize; ++src) {
        // Get size and start of the source process
        if (m_sqmatarr.m_prank == src) {
            chunksize = m_sqmatarr.m_mastsize_flat * nmsq;
            chunkstart = m_sqmatarr.m_mastdispl_flat * nmsq;
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
*/

// Type for partitioning only the first dimension
template<typename SqMatArrType>
class MastDim0Part {
public:
    explicit MastDim0Part(SqMatArrType& sqmatarr) : m_sqmatarr(sqmatarr) {
        mostEvenPart(sqmatarr.dim0(), sqmatarr.procSize(), sqmatarr.procRank(), m_size, m_displ);  // Partition only the first dimension
    }
    MastDim0Part(const MastDim0Part&) = default;  // Default copy and move constructors are good for reference member
    MastDim0Part(MastDim0Part&&) = default;
    MastDim0Part& operator=(const MastDim0Part&) = delete;
    MastDim0Part& operator=(MastDim0Part&&) = delete;
    ~MastDim0Part() = default;
    
    Eigen::Index dim0() const {return m_size;}
    Eigen::Index displ() const {return m_displ;}
    Eigen::Index dim1() const {return m_sqmatarr.dim1();}
    Eigen::Index dimm() const {return m_sqmatarr.dimm();}
    
    typename SqMatArrType::ColsBlockXpr operator()(void) {  // Colume number is already dynamic due to mastsize_dim0
        return m_sqmatarr.m_data.middleCols(m_displ * m_sqmatarr.dim1() * m_sqmatarr.dimm(), m_size * m_sqmatarr.dim1() * m_sqmatarr.dimm());
    }
    typename SqMatArrType::ConstColsBlockXpr operator()(void) const {  // Colume number is already dynamic due to mastsize_dim0
        return m_sqmatarr.m_data.middleCols(m_displ * m_sqmatarr.dim1() * m_sqmatarr.dimm(), m_size * m_sqmatarr.dim1() * m_sqmatarr.dimm());
    }
    typename SqMatArrType::template NColsBlockXpr<SqMatArrType::Dim1mAtCompileTime> atDim0(const Eigen::Index i0) {
        assert(i0 < m_size);
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::Dim1mAtCompileTime>((i0 + m_displ) * m_sqmatarr.dim1() * m_sqmatarr.dimm(),
                                                                                       m_sqmatarr.dim1() * m_sqmatarr.dimm());
    }
    typename SqMatArrType::template ConstNColsBlockXpr<SqMatArrType::Dim1mAtCompileTime> atDim0(const Eigen::Index i0) const {
        assert(i0 < m_size);
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::Dim1mAtCompileTime>((i0 + m_displ) * m_sqmatarr.dim1() * m_sqmatarr.dimm(),
                                                                                       m_sqmatarr.dim1() * m_sqmatarr.dimm());
    }
    // Access each dim1 slice
    auto atDim1(const Eigen::Index i1) {
        assert(i1 < m_sqmatarr.dim1());
        //std::cout << "non-const version" << std::endl;
        return m_sqmatarr.m_data(Eigen::all, typename SqMatArrType::slice_dim1{m_size, m_sqmatarr.dim1(), m_sqmatarr.dimm(), i1 + m_displ * m_sqmatarr.dim1()});
    }
    const auto atDim1(const Eigen::Index i1) const {
        assert(i1 < m_sqmatarr.dim1());
        //std::cout << "const version" << std::endl;
        return m_sqmatarr.m_data(Eigen::all, typename SqMatArrType::slice_dim1{m_size, m_sqmatarr.dim1(), m_sqmatarr.dimm(), i1 + m_displ * m_sqmatarr.dim1()});
    }
    // Return a view of the matrix whose rows are vectors along dim1 of every unit matrix element, at a given dim0 index
    auto dim1RowVecsAtDim0(const Eigen::Index i0) {
        assert(i0 < m_size);
        //return this->m_data(im0, Eigen::seqN(i0 * this->dim1() * this->dimm() + im1, this->dim1(), this->dimm()));
        return this->atDim0(i0).reshaped(m_sqmatarr.dimm() * m_sqmatarr.dimm(), m_sqmatarr.dim1());
    }
    const auto dim1RowVecsAtDim0(const Eigen::Index i0) const {
        assert(i0 < m_size);
        //return this->m_data(im0, Eigen::seqN(i0 * this->dim1() * this->dimm() + im1, this->dim1(), this->dimm()));
        return this->atDim0(i0).reshaped(m_sqmatarr.dimm() * m_sqmatarr.dimm(), m_sqmatarr.dim1());
    }
    typename SqMatArrType::template NColsBlockXpr<SqMatArrType::DimmAtCompileTime> operator()(const Eigen::Index i0, const Eigen::Index i1) {
        assert(i0 < m_size && i1 < m_sqmatarr.dim1());
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::DimmAtCompileTime>(((i0 + m_displ) * m_sqmatarr.dim1() + i1) * m_sqmatarr.dimm(), m_sqmatarr.dimm());
    }
    typename SqMatArrType::template ConstNColsBlockXpr<SqMatArrType::DimmAtCompileTime> operator()(const Eigen::Index i0, const Eigen::Index i1) const {
        assert(i0 < m_size && i1 < m_sqmatarr.dim1());
        return m_sqmatarr.m_data.template middleCols<SqMatArrType::DimmAtCompileTime>(((i0 + m_displ) * m_sqmatarr.dim1() + i1) * m_sqmatarr.dimm(), m_sqmatarr.dimm());
    }
    typename SqMatArrType::Scalar& operator()(const Eigen::Index i0, const Eigen::Index i1, const Eigen::Index im0, const Eigen::Index im1) {
        assert(i0 < m_size && i1 < m_sqmatarr.dim1() && im0 < m_sqmatarr.dimm() && im1 < m_sqmatarr.dimm());
        return m_sqmatarr.m_data(im0, ((i0 + m_displ) * m_sqmatarr.dim1() + i1) * m_sqmatarr.dimm() + im1);
    }
    typename SqMatArrType::Scalar operator()(const Eigen::Index i0, const Eigen::Index i1, const Eigen::Index im0, const Eigen::Index im1) const {
        assert(i0 < m_size && i1 < m_sqmatarr.dim1() && im0 < m_sqmatarr.dimm() && im1 < m_sqmatarr.dimm());
        return m_sqmatarr.m_data(im0, ((i0 + m_displ) * m_sqmatarr.dim1() + i1) * m_sqmatarr.dimm() + im1);
    }
    
    // Sum-reduce _data on all local processes and the results are separately held in the mastered partition on each process
    void sum2mastPart() {
        const Eigen::Index dim1nmsq = m_sqmatarr.dim1() * m_sqmatarr.dimm() * m_sqmatarr.dimm();
        m_sqmatarr.sum2mastPart(m_size * dim1nmsq, m_displ * dim1nmsq);
    }
    // Gather each piece governed by each local process and broadcast the result to all processes.
    // Note every process holds the full-sized _data, but it is assumed that each process governs
    // a piece of _data. There is no need to really scatter _data to processes and reallocate
    // _data every time when entering DMFT equations from QMC run, because _data upon exiting QMC
    // run on every process is intact.
    void allGather() {
        const Eigen::Index dim1nmsq = m_sqmatarr.dim1() * m_sqmatarr.dimm() * m_sqmatarr.dimm();
        m_sqmatarr.allGather(m_size * dim1nmsq, m_displ * dim1nmsq);
    }
    
private:
    SqMatArrType& m_sqmatarr;
    Eigen::Index m_size, m_displ;
};

/*
template<typename SqMatArrType>
void MastDim0Part<SqMatArrType>::sum2mastPart() {
    static_assert(std::is_same<typename SqMatArrType::Scalar, double>::value || std::is_same<typename SqMatArrType::Scalar, std::complex<double> >::value);
    if (m_sqmatarr.m_is_inter) throw std::invalid_argument("MPI communicator is an intercommunicator prohibiting in-place Reduce!");
    const Eigen::Index dim1nmsq = m_sqmatarr.dim1() * m_sqmatarr.dimm() * m_sqmatarr.dimm();
    Eigen::Index chunksize, chunkstart;
    for (int dest = 0; dest < m_sqmatarr.m_psize; ++dest) {
        // Get size and start of the destination process
        if (m_sqmatarr.m_prank == dest) {
            chunksize = m_sqmatarr.m_mastsize_dim0 * dim1nmsq;
            chunkstart = m_sqmatarr.m_mastdispl_dim0 * dim1nmsq;
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
    const Eigen::Index dim1nmsq = m_sqmatarr.dim1() * m_sqmatarr.dimm() * m_sqmatarr.dimm();
    Eigen::Index chunksize, chunkstart;
    for (int src = 0; src < m_sqmatarr.m_psize; ++src) {
        // Get size and start of the source process
        if (m_sqmatarr.m_prank == src) {
            chunksize = m_sqmatarr.m_mastsize_dim0 * dim1nmsq;
            chunkstart = m_sqmatarr.m_mastdispl_dim0 * dim1nmsq;
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
*/

/*****************************************************************************************************
Formal data wrapper class template
*******************************************************************************************************/

template<typename _Scalar, int _n0, int _n1, int _nm>
class SqMatArray : public SqMatArrayStorage<_Scalar, _n0, _n1, _nm> {
    template<typename SqMatArrType>
    friend class MastFlatPart;
    template<typename SqMatArrType>
    friend class MastDim0Part;
    
public:
    typedef _Scalar Scalar;
    static constexpr int DimmAtCompileTime = _nm;
    static constexpr int Dim1mAtCompileTime = _n1 == Eigen::Dynamic || _nm == Eigen::Dynamic ? Eigen::Dynamic : _n1 * _nm;
    static constexpr int Dim01mAtCompileTime = _n0 == Eigen::Dynamic || _n1 == Eigen::Dynamic || _nm == Eigen::Dynamic ? Eigen::Dynamic : _n0 * _n1 * _nm;
    typedef typename Eigen::DenseBase<typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType>::ColsBlockXpr ColsBlockXpr;
    typedef typename Eigen::DenseBase<typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType>::ConstColsBlockXpr ConstColsBlockXpr;
    // Alias templates. Below templates accept _nm being fixed size or dynamic, and in the former case the runtime size must equal _nm
    template <int N>
    using NColsBlockXpr = typename Eigen::DenseBase<typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType>::template NColsBlockXpr<N>::Type;
    template <int N>
    using ConstNColsBlockXpr = typename Eigen::DenseBase<typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType>::template ConstNColsBlockXpr<N>::Type;
    
    explicit SqMatArray(const MPI_Comm& comm = MPI_COMM_SELF) : m_comm(comm) {
        MPI_Comm_size(comm, &m_psize); MPI_Comm_rank(comm, &m_prank); MPI_Comm_test_inter(comm, &m_is_inter);
    }
    SqMatArray(const Eigen::Index n0, const Eigen::Index n1, const Eigen::Index nm, const MPI_Comm& comm = MPI_COMM_SELF) :
    SqMatArrayStorage<_Scalar, _n0, _n1, _nm>(n0, n1, nm), m_comm(comm) {
        MPI_Comm_size(comm, &m_psize); MPI_Comm_rank(comm, &m_prank); MPI_Comm_test_inter(comm, &m_is_inter);
    }
    SqMatArray(const SqMatArray&) = default;
    SqMatArray(SqMatArray&&) = default;
    SqMatArray& operator=(const SqMatArray&) = default;
    SqMatArray& operator=(SqMatArray&&) = default;
    ~SqMatArray() = default;
    
    const MPI_Comm& mpiComm(void) const {return m_comm;}
    void mpiComm(const MPI_Comm& comm) {
        m_comm = comm;
        MPI_Comm_size(comm, &m_psize); MPI_Comm_rank(comm, &m_prank); MPI_Comm_test_inter(comm, &m_is_inter);
    }
    int procSize() const {return m_psize;}
    int procRank() const {return m_prank;}
    
    // Expose the underlying Eigen::Matrix object for global manipulation (should not do resize)
    //typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType& operator()() {return this->m_data;}
    //const typename SqMatArrayStorage<_Scalar, _n0, _n1, _nm>::DataType& operator()() const {return this->m_data;}
    // Return Eigen Block object meant to prevent silent resizing; this is also more consistent with other accessing methods
    NColsBlockXpr<Dim01mAtCompileTime> operator()(void) {return this->m_data.template leftCols<Dim01mAtCompileTime>(this->dim0() * this->dim1() * this->dimm());}
    ConstNColsBlockXpr<Dim01mAtCompileTime> operator()(void) const {return this->m_data.template leftCols<Dim01mAtCompileTime>(this->dim0() * this->dim1() * this->dimm());}
    
    // Just sequentially access each unit matrix, which becomes intuitive when _n0 = 1 or _n1 = 1.
    NColsBlockXpr<DimmAtCompileTime> operator[](const Eigen::Index i) {
        assert(i < this->size());
        return this->m_data.template middleCols<_nm>(i * this->dimm(), this->dimm());
    }
    ConstNColsBlockXpr<DimmAtCompileTime> operator[](const Eigen::Index i) const {
        assert(i < this->size());
        return this->m_data.template middleCols<_nm>(i * this->dimm(), this->dimm());
    }
    
    // Access each dim0 block
    NColsBlockXpr<Dim1mAtCompileTime> atDim0(const Eigen::Index i0) {
        assert(i0 < this->dim0());
        return this->m_data.template middleCols<Dim1mAtCompileTime>(i0 * this->dim1() * this->dimm(), this->dim1() * this->dimm());
    }
    ConstNColsBlockXpr<Dim1mAtCompileTime> atDim0(const Eigen::Index i0) const {
        assert(i0 < this->dim0());
        return this->m_data.template middleCols<Dim1mAtCompileTime>(i0 * this->dim1() * this->dimm(), this->dim1() * this->dimm());
    }
    
    // Access each dim1 slice
    auto atDim1(const Eigen::Index i1) {
        assert(i1 < this->dim1());
        //std::cout << "non-const version" << std::endl;
        return this->m_data(Eigen::all, slice_dim1{this->dim0(), this->dim1(), this->dimm(), i1});
    }
    const auto atDim1(const Eigen::Index i1) const {
        assert(i1 < this->dim1());
        //std::cout << "const version" << std::endl;
        return this->m_data(Eigen::all, slice_dim1{this->dim0(), this->dim1(), this->dimm(), i1});
    }
    
    // Return a view of the matrix whose rows are vectors along dim1 of every unit matrix element, at a given dim0 index
    auto dim1RowVecsAtDim0(const Eigen::Index i0) {
        assert(i0 < this->dim0());
        //return this->m_data(im0, Eigen::seqN(i0 * this->dim1() * this->dimm() + im1, this->dim1(), this->dimm()));
        return this->atDim0(i0).reshaped(this->dimm() * this->dimm(), this->dim1());
    }
    const auto dim1RowVecsAtDim0(const Eigen::Index i0) const {
        assert(i0 < this->dim0());
        //return this->m_data(im0, Eigen::seqN(i0 * this->dim1() * this->dimm() + im1, this->dim1(), this->dimm()));
        return this->atDim0(i0).reshaped(this->dimm() * this->dimm(), this->dim1());
    }
    
    // Overload operator () to intuitively access each unit matrix
    NColsBlockXpr<_nm> operator()(const Eigen::Index i0, const Eigen::Index i1) {  // Non-const version
        assert(i0 < this->dim0() && i1 < this->dim1());
        return this->m_data.template middleCols<_nm>((i0 * this->dim1() + i1) * this->dimm(), this->dimm());
    }
    ConstNColsBlockXpr<_nm> operator()(const Eigen::Index i0, const Eigen::Index i1) const {   // Const version
        assert(i0 < this->dim0() && i1 < this->dim1());
        return this->m_data.template middleCols<_nm>((i0 * this->dim1() + i1) * this->dimm(), this->dimm());
    }
    
    // Directly access element
    _Scalar& operator()(const Eigen::Index i0, const Eigen::Index i1, const Eigen::Index im0, const Eigen::Index im1) {
        assert(i0 < this->dim0() && i1 < this->dim1() && im0 < this->dimm() && im1 < this->dimm());
        return this->m_data(im0, (i0 * this->dim1() + i1) * this->dimm() + im1);
    }
    _Scalar operator()(const Eigen::Index i0, const Eigen::Index i1, const Eigen::Index im0, const Eigen::Index im1) const {
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
    
    void allSum();   // allSum-reduce the whole m_data on all local processes
    void broadcast(const int rank);   // Broadcast the whole m_data on process "rank" to other processes
    
protected:
    // Need these MPI related variables because this class itself needs some MPI operations such as all-sum
    MPI_Comm m_comm;
    int m_psize, m_prank, m_is_inter;
    
    struct slice_dim1 {
        Eigen::Index size() const {return n0 * nm;}
        Eigen::Index operator[](Eigen::Index i) const {return i % nm + (i1 + (i / nm) * n1) * nm;}
        Eigen::Index n0, n1, nm, i1;
    };
    
    // Helper functions for friend classes
    void sum2mastPart(const int localsize, const int localdispl);
    void allGather(const int localsize, const int localdispl);
};

template<typename _Scalar, int _n0, int _n1, int _nm>
void SqMatArray<_Scalar, _n0, _n1, _nm>::allSum() {
    static_assert(std::is_same<_Scalar, double>::value || std::is_same<_Scalar, std::complex<double> >::value);
    if (m_is_inter) throw std::invalid_argument("MPI communicator is an intercommunicator prohibiting in-place Allreduce!");
    if constexpr (std::is_same<_Scalar, double>::value)
        MPI_Allreduce(MPI_IN_PLACE, this->m_data.data(), this->m_data.size(), MPI_DOUBLE, MPI_SUM, m_comm);
    else if (std::is_same<_Scalar, std::complex<double> >::value)
        MPI_Allreduce(MPI_IN_PLACE, this->m_data.data(), this->m_data.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, m_comm);
}

template<typename _Scalar, int _n0, int _n1, int _nm>
void SqMatArray<_Scalar, _n0, _n1, _nm>::broadcast(const int rank) {
    static_assert(std::is_same<_Scalar, double>::value || std::is_same<_Scalar, std::complex<double> >::value);
    if constexpr (std::is_same<_Scalar, double>::value)
        MPI_Bcast(this->m_data.data(), this->m_data.size(), MPI_DOUBLE, rank, m_comm);
    else if (std::is_same<_Scalar, std::complex<double> >::value)
        MPI_Bcast(this->m_data.data(), this->m_data.size(), MPI_DOUBLE_COMPLEX, rank, m_comm);
}

template<typename _Scalar, int _n0, int _n1, int _nm>
void SqMatArray<_Scalar, _n0, _n1, _nm>::sum2mastPart(const int localsize, const int localdispl) {
    static_assert(std::is_same<_Scalar, double>::value || std::is_same<_Scalar, std::complex<double> >::value);
    if (m_is_inter) throw std::invalid_argument("MPI communicator is an intercommunicator prohibiting in-place Reduce!");
    int chunksize, chunkstart;
    for (int dest = 0; dest < m_psize; ++dest) {
        // Broadcast size and start of the destination process to all processes
        if (m_prank == dest) {
            chunksize = localsize;
            chunkstart = localdispl;
        }
        MPI_Bcast(&chunksize, 1, MPI_INT, dest, m_comm);
        MPI_Bcast(&chunkstart, 1, MPI_INT, dest, m_comm);
        
        if (m_prank == dest) {
            if constexpr (std::is_same<_Scalar, double>::value)
                MPI_Reduce(MPI_IN_PLACE, this->m_data.data() + chunkstart, chunksize, MPI_DOUBLE, MPI_SUM, dest, m_comm);
            else if (std::is_same<_Scalar, std::complex<double> >::value)
                MPI_Reduce(MPI_IN_PLACE, this->m_data.data() + chunkstart, chunksize, MPI_DOUBLE_COMPLEX, MPI_SUM, dest, m_comm);
        }
        else {
            if constexpr (std::is_same<_Scalar, double>::value)
                MPI_Reduce(this->m_data.data() + chunkstart, this->m_data.data() + chunkstart, chunksize, MPI_DOUBLE, MPI_SUM, dest, m_comm);
            else if (std::is_same<_Scalar, std::complex<double> >::value)
                MPI_Reduce(this->m_data.data() + chunkstart, this->m_data.data() + chunkstart, chunksize, MPI_DOUBLE_COMPLEX, MPI_SUM, dest, m_comm);
        }
    }
}

template<typename _Scalar, int _n0, int _n1, int _nm>
void SqMatArray<_Scalar, _n0, _n1, _nm>::allGather(const int localsize, const int localdispl) {
    static_assert(std::is_same<_Scalar, double>::value || std::is_same<_Scalar, std::complex<double> >::value);
    int *recvcounts = new int[m_psize];
    int *displs = new int[m_psize];
    recvcounts[m_prank] = localsize;
    displs[m_prank] = localdispl;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, m_comm);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, displs, 1, MPI_INT, m_comm);
    if constexpr (std::is_same<_Scalar, double>::value)
        MPI_Allgatherv(MPI_IN_PLACE, recvcounts[m_prank], MPI_DOUBLE, this->m_data.data(), recvcounts, displs, MPI_DOUBLE, m_comm);
    else if (std::is_same<_Scalar, std::complex<double> >::value)
        MPI_Allgatherv(MPI_IN_PLACE, recvcounts[m_prank], MPI_DOUBLE_COMPLEX, this->m_data.data(), recvcounts, displs, MPI_DOUBLE_COMPLEX, m_comm);
    delete[] recvcounts;
    delete[] displs;
}







// Overload the insertion operator
template<typename Scalar, int n0, int n1, int nm>
std::ostream& operator<<(std::ostream& os, const SqMatArray<Scalar, n0, n1, nm>& sqmats) {
//    int charspernum;
//    std::array<Eigen::Index, 2> i2d;
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
    std::ostringstream ostr;
    Eigen::Index i0, i1, im0, im1;
    const Eigen::Index size = sqmats().size();
    int width = 0;
    
    // Get maximum width
    ostr.copyfmt(os);
    for (i0 = 0; i0 < size; ++i0) {
        if constexpr (std::is_same<Scalar, std::complex<double> >::value) {
            ostr << sqmats()(i0).real();
            width = std::max<int>(width, ostr.str().length());
            ostr.str(std::string());
            ostr.clear();
            ostr << sqmats()(i0).imag();
            width = std::max<int>(width, ostr.str().length());
            ostr.str(std::string());
            ostr.clear();
        }
        else {
            ostr << sqmats()(i0);
            width = std::max<int>(width, ostr.str().length());
            ostr.str(std::string());
            ostr.clear();
        }
    }
    
    for (i1 = 0; i1 < sqmats.dim1(); ++i1) {
        for (i0 = 0; i0 < sqmats.dim0(); ++i0) {
            for (im1 = 0; im1 < sqmats.dimm(); ++im1) {
                for (im0 = 0; im0 < sqmats.dimm(); ++im0) {
                    if constexpr (std::is_same<Scalar, std::complex<double> >::value) os << std::setw(width) << sqmats(i0, i1, im0, im1).real() << " "
                        << std::setw(width) << sqmats(i0, i1, im0, im1).imag() << "  ";
                    else os << std::setw(width) << sqmats(i0, i1, im0, im1) << "  ";
                }
            }
        }
        if (i1 < sqmats.dim1() - 1) os << std::endl;
    }
    return os;
}

/*
// read one field of numerics including inf and nan
template <typename Scalar>
std::istream& read_numerics(std::istream& is, Scalar& x) {
    std::string word;
    is >> word;
    auto [ptr, ec]{std::from_chars(word.data(), word.data() + word.size(), x)};
    if (ec == std::errc::invalid_argument) {
        if constexpr (std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value || std::is_same<Scalar, long double>::value) {
            if (word == "inf" || word == "+inf") x = std::numeric_limits<double>::infinity();
            else if (word == "-inf") x = -std::numeric_limits<double>::infinity();
            else if (word == "nan" || word == "+nan") x = std::nan("readin");
            else if (word == "-nan") x = -std::nan("readin");
            else is.setstate(std::ios::failbit);
        }
        else is.setstate(std::ios::failbit);
    }
    else if (ec == std::errc::result_out_of_range) is.setstate(std::ios::failbit);
    return is;
}
*/

// Overload the extraction operator
template<typename Scalar, int n0, int n1, int nm>
std::istream& operator>>(std::istream& is, SqMatArray<Scalar, n0, n1, nm>& sqmats) {
    Eigen::Index i0, i1, im0, im1;
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

// Overload extraction operator for filling Eigen objects with istream; support allocating A according to the size of
// input stream (from current position to end) if A hasn't been allocated
template <typename Derived>
std::istream& operator>>(std::istream& is, const Eigen::DenseBase<Derived>& A) {
    static_assert(!std::is_same<typename Derived::Scalar, std::complex<double> >::value,
                  "Extraction operator from istream to Eigen objects is currently not applicable to complex scalar type");
    
    Eigen::DenseBase<Derived>& A_ = const_cast<Eigen::DenseBase<Derived>&>(A);
    
    if (A.size() == 0) {  // A hasn't been allocated, then resize A to take all numbers from the current position to end in input stream
        auto state_backup = is.rdstate();
        auto pos_backup = is.tellg();
        std::string line;
        std::istringstream line0;
        std::getline(is, line);
        line0.str(line);
        Eigen::Index n_cols = std::distance(std::istream_iterator<std::string>(line0), std::istream_iterator<std::string>());
        Eigen::Index n_rows = 0;
        do {
            if (!line.empty()) ++n_rows;
        } while (std::getline(is, line));
        is.clear(state_backup);  // Method clear is to overwritten the current flags; method setstate is not to overwritten but combine
        is.seekg(pos_backup);
        
        A_.derived().resize(n_rows, n_cols);
    }
    
    std::string word;
    for (Eigen::Index i = 0; i < A_.rows(); ++i)
        for (Eigen::Index j = 0; j < A_.cols(); ++j) {
            if constexpr (std::is_same<typename Derived::Scalar, double>::value) {
                is >> word;
                try {
                    A_(i, j) = std::stod(word);
                }
                catch (std::invalid_argument& e) {
                    if (word == "inf" || word == "+inf") A_(i, j) = std::numeric_limits<double>::infinity();
                    else if (word == "-inf") A_(i, j) = -std::numeric_limits<double>::infinity();
                    else if (word == "nan" || word == "+nan") A_(i, j) = std::nan("readin");
                    else if (word == "-nan") A_(i, j) = -std::nan("readin");
                    else {
                        is.setstate(std::ios::failbit);
                        return is;
                    }
                }
            }
            else is >> A_(i, j);
        }
    return is;
}




typedef SqMatArray<double, 2, 1, Eigen::Dynamic> SqMatArray21Xd;
typedef SqMatArray<std::complex<double>, 2, 1, Eigen::Dynamic> SqMatArray21Xcd;
typedef SqMatArray<double, 2, 2, Eigen::Dynamic> SqMatArray22Xd;
typedef SqMatArray<std::complex<double>, 2, 2, Eigen::Dynamic> SqMatArray22Xcd;
typedef SqMatArray<std::complex<double>, 2, 3, Eigen::Dynamic> SqMatArray23Xcd;
typedef SqMatArray<double, 2, Eigen::Dynamic, Eigen::Dynamic> SqMatArray2XXd;
typedef SqMatArray<std::complex<double>, 2, Eigen::Dynamic, Eigen::Dynamic> SqMatArray2XXcd;
typedef SqMatArray<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> SqMatArrayXXXd;
typedef SqMatArray<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> SqMatArrayXXXcd;

typedef Eigen::Array<Eigen::Index, Eigen::Dynamic, 1> ArrayXindex;


template <typename T>
void printData(const std::string& fname, const T& data, const int precision = 6) {
    std::ofstream fout(fname, std::fstream::out | std::fstream::trunc);
    fout << std::setprecision(precision);
    if (fout.is_open()) {
        fout << data;
        fout.close();
    }
    else std::cout << "Unable to open file" << std::endl;
}

template <typename T>
void loadData(const std::string& fname, T& A) {
    std::ifstream fin(fname);
    if (fin.is_open()) {
        fin >> A;
        fin.close();
    }
    else std::cout << "Unable to open file" << std::endl;
}


#endif /* gf_data_wrapper_hpp */
