//
//  bare_hamiltonian.cpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#include <iostream>
#include <cmath>
#include "bare_hamiltonian.hpp"



void BareHamiltonian::kVecAtIndex(std::size_t ik, Eigen::VectorXd& k) const {
//    if (_K.rows() == 1) k = static_cast<double>(ik) / _nk(0) * _K.col(0);
//    else if (_K.rows() == 2) {
//        const std::size_t ix = ik / _nk(1);
//        const std::size_t iy = ik % _nk(1);
//        k = static_cast<double>(ix) / _nk(0) * _K.col(0) + static_cast<double>(iy) / _nk(1) * _K.col(1);
//    }
//    else if (_K.rows() == 3) {
//        const std::size_t nk12 = _nk(1) * _nk(2);
//        const std::size_t ix = ik / nk12;
//        const std::size_t iy = (ik % nk12) / _nk(2);
//        const std::size_t iz = (ik % nk12) % _nk(2);
//        k = static_cast<double>(ix) / _nk(0) * _K.col(0) + static_cast<double>(iy) / _nk(1) * _K.col(1) + static_cast<double>(iz) / _nk(2) * _K.col(2);
//    }
    Eigen::VectorXd kfrac(m_K.cols());
    std::size_t nkv = m_nk.prod();
    for (std::size_t n = 0; n < m_K.cols(); ++n) {
        nkv /= m_nk(n);
        kfrac(n) = static_cast<double>(ik / nkv) / m_nk(n);
        ik %= nkv;
    }
    k = (m_K * kfrac.asDiagonal()).rowwise().sum();
}

void BareHamiltonian::setMPIcomm(const MPI_Comm& comm) {
    m_comm = comm;
    MPI_Comm_size(comm, &m_psize);
    MPI_Comm_rank(comm, &m_prank);
}

void BareHamiltonian::constructHamiltonian(const Eigen::VectorXd& k, Eigen::MatrixXcd& H) const {
    throw std::domain_error("constructHamiltonian() has not been defined!");
}

void BareHamiltonian::constructFermiVelocities(const int coord, const Eigen::VectorXd& k, Eigen::MatrixXcd& v) const {
    throw std::domain_error("constructFermiVelocities() has not been defined!");
}

void BareHamiltonian::computeDOS(const std::size_t nbins) {
    int is_inter;
    MPI_Comm_test_inter(m_comm, &is_inter);
    if (is_inter) throw std::invalid_argument( "MPI communicator is an intercommunicator prohibiting in-place Allreduce!" );
    if (m_bands.size() == 0) throw std::range_error("Cannot compute DOS because bands have not been computed!");
    
    std::size_t ie, ib, ik;
    const double binsize = (m_erange[1] - m_erange[0]) / nbins;
    
    m_dos.resize(nbins);
    m_dos.setZero();
    
    for (ik = 0; ik < m_klocalsize; ++ik) {
        for (ib = 0; ib < m_bands.rows(); ++ib) {
            ie = std::min(static_cast<std::size_t>((m_bands(ib, ik) - m_erange[0]) / binsize), nbins - 1);
            m_dos[ie] += 1.0;
        }
    }
    
    MPI_Allreduce(MPI_IN_PLACE, m_dos.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, m_comm);  // Complete DOS is needed by every process
    m_dos /= m_nk.prod() * binsize;   // DOS is per unit cell per energy, for its use in k-space Fourier inversion of the lattice Green's function
}
