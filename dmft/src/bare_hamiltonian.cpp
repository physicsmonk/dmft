//
//  bare_hamiltonian.cpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#include <iostream>
#include <cmath>
#include "bare_hamiltonian.hpp"


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

void BareHamiltonian::computeDOS(const Eigen::Index nbins) {
    int is_inter;
    MPI_Comm_test_inter(m_comm, &is_inter);
    if (is_inter) throw std::invalid_argument( "MPI communicator is an intercommunicator prohibiting in-place Allreduce!" );
    if (m_bands.rows() == 0) throw std::range_error("Cannot compute DOS because bands have not been computed!");
    
    Eigen::Index ie, kid;
    const double binsize = (m_erange[1] - m_erange[0]) / nbins;
    
    m_dos = Eigen::ArrayX2d::Zero(nbins, 2);
    m_dos.col(0).setLinSpaced(nbins, m_erange[0] + binsize * 0.5, m_erange[1] - binsize * 0.5);   // Record energies
    
    for (Eigen::Index kidlocal = 0; kidlocal < m_klocalsize; ++kidlocal) {
        kid = kidlocal + m_klocalstart;
        for (Eigen::Index ib = m_a.rows(); ib < m_bands.rows(); ++ib) {
            ie = std::min(static_cast<Eigen::Index>((m_bands(ib, kid) - m_erange[0]) / binsize), nbins - 1);
            m_dos(ie, 1) += 1.0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, m_dos.col(1).data(), nbins, MPI_DOUBLE, MPI_SUM, m_comm);  // Complete DOS is needed by every process
    m_dos.col(1) /= m_nk.prod() * binsize;   // DOS is per unit cell per energy, for its use in k-space Fourier inversion of the lattice Green's function
}
