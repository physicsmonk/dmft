//
//  self_consistency.cpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#include <iostream>
#include <cmath>
#include <Eigen/LU>
#include "self_consistency.hpp"

using namespace std::complex_literals;


DMFTIterator::DMFTIterator(std::shared_ptr<const BareHamiltonian> H0, std::shared_ptr<BareGreenFunction> Gbath, std::shared_ptr<const GreenFunction> Gimp) :
m_ptr2H0(H0), m_ptr2Gbath(Gbath), m_ptr2Gimp(Gimp), m_Glat(2, Gimp->freqCutoff() + 1, Gimp->nSites(), Gimp->fourierCoeffs().mpiCommunicator()),
m_selfen_dyn(2, Gimp->freqCutoff() + 1, Gimp->nSites(), Gimp->fourierCoeffs().mpiCommunicator()), m_selfen_static(2, 1, Gimp->nSites()),
m_selfen_moms(2, 3, Gimp->nSites()), m_selfen_var(2, Gimp->freqCutoff() + 1, Gimp->nSites(), Gimp->fourierCoeffs().mpiCommunicator()), m_iter(0) {
    // Default parameters
    parameters["G0 update step size"] = 1.0;
    parameters["convergence type"] = std::string("Gimp_Glat_max_error");  // Or "Gimp_Glat_average_error", "G0_..."
    parameters["convergence criterion"] = 0.005;
    parameters["local correlation"] = false;
    
    // Compute moments of bath Green's function, which only depend on bare Hamiltonian
    // and thus only need to be calculated once when bare Hamiltonian passed in.
    m_ptr2Gbath->computeMoments(*m_ptr2H0);
}

// Update the bath Green's function using the current lattice Green's function and self-energy
void DMFTIterator::updateBathGF() {
    //++m_iter;
    
    // First record old bath Green's function in imaginary-time space for corresponding cases
    const auto convergtype = std::any_cast<std::string>(parameters.at("convergence type"));
    if (convergtype == "G0_average_error" || convergtype == "G0_max_error") m_G0old = m_ptr2Gbath->valsOnTauGrid();
    
    auto stepsize = std::any_cast<double>(parameters.at("G0 update step size"));
    if (stepsize < 0 || stepsize > 1) throw std::invalid_argument("Step size for updating bath Green's function must be in [0, 1]!");
    
    auto Gbathmastpart = m_ptr2Gbath->fourierCoeffs().mastFlatPart();
    auto Glatmastpart = m_Glat.mastFlatPart();
    std::array<std::size_t, 2> so;
    
    if (m_iter == 0) {
        stepsize = 1.0;  // This means that at zero step (initialization), directly initialize bath Green's function
        Gbathmastpart().setZero();
    }
    
    if (m_ptr2H0->type() == "bethe") {
        if (m_ptr2Gimp->nSites() != 1) throw std::range_error("Number of sites must be 1 for Bethe lattice (with semicircular DOS)!");
        const std::complex<double> t = m_ptr2H0->hopMatElem(0);
        std::complex<double> iwu(m_ptr2H0->chemPot(), 0.0);
        
        for (std::size_t i = 0; i < Glatmastpart.size(); ++i) {
            so = Glatmastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
            iwu.imag(m_ptr2Gimp->matsubFreqs()(so[1]));
            Gbathmastpart(i, 0, 0) = (1.0 - stepsize) * Gbathmastpart(i, 0, 0) + stepsize / (-iwu - (t * t) * Glatmastpart(i, 0, 0));  // Number of sites is 1
        }
    }
    else if (m_ptr2H0->type() == "bethe_dimer") {
        if (m_ptr2Gimp->nSites() != 2) throw std::range_error("Number of sites must be 2 for dimer Hubbard model with semicircular density of states!");
        const std::complex<double> t = m_ptr2H0->hopMatElem(0);
        const std::complex<double> tz = m_ptr2H0->hopMatElem(1);
        Eigen::Matrix2cd zeta;
        
        zeta << m_ptr2H0->chemPot(), tz,
                tz,             m_ptr2H0->chemPot();
        
        for (std::size_t i = 0; i < Glatmastpart.size(); ++i) {
            so = Glatmastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
            zeta(0, 0).imag(m_ptr2Gimp->matsubFreqs()(so[1]));
            zeta(1, 1).imag(m_ptr2Gimp->matsubFreqs()(so[1]));
            Gbathmastpart[i] = (1.0 - stepsize) * Gbathmastpart[i] + stepsize * (-zeta - (t * t) * Glatmastpart[i]).inverse();
        }
    }
    else {
        auto selfen_dyn_mastpart = m_selfen_dyn.mastFlatPart();
        for (std::size_t i = 0; i < Glatmastpart.size(); ++i) {
            so = selfen_dyn_mastpart.global2dIndex(i);
            Gbathmastpart[i] = (1.0 - stepsize) * Gbathmastpart[i] + stepsize * (Glatmastpart[i].inverse() - selfen_dyn_mastpart[i] - m_selfen_static[so[0]]).inverse();
        }
    }
    
    // Do all-gather because every process needs full access to the Fourier coefficients of the Bath Green's function
    // for measurements in the next QMC run
    Gbathmastpart.allGather();
    // Fourier inversion does not require the Fourier coefficients have been all-gathered. And it already spreads the
    // complete inversion result to all processes for the next QMC run.
    m_ptr2Gbath->invFourierTrans();   // Spline built using already-calculated high-frequency expansion
}

// Calculate static part and moments of self-energy, remember the used nonstandard definition of Green's function
void DMFTIterator::computeSelfEnStatMoms() {
    for (int s = 0; s < 2; ++s) {
        m_selfen_static[s] = m_ptr2Gbath->moments()(s, 1) - m_ptr2Gimp->moments()(s, 1);  // This is the static part
        m_selfen_moms(s, 0).noalias() = -m_ptr2Gimp->moments()(s, 2) - m_ptr2Gimp->moments()(s, 1) * m_ptr2Gimp->moments()(s, 1)
        + m_ptr2Gbath->moments()(s, 2) + m_ptr2Gbath->moments()(s, 1) * m_ptr2Gbath->moments()(s, 1);
        // Second and third coefficients to be implemented
    }
}

// Approximate self-energy from the solved impurity problem
void DMFTIterator::approxSelfEnergy() {
    auto selfen_dyn_mastpart = m_selfen_dyn.mastFlatPart();
    auto selfen_var_mastpart = m_selfen_var.mastFlatPart();
    const auto Gimpmastpart = m_ptr2Gimp->fourierCoeffs().mastFlatPart();
    const auto Gimpvarmastpart = m_ptr2Gimp->fCoeffsVar().mastFlatPart();
    auto Gbathmastpart = m_ptr2Gbath->fourierCoeffs().mastFlatPart();
    std::array<std::size_t, 2> so;
    Eigen::MatrixXd tmp;
    computeSelfEnStatMoms();
    for (std::size_t i = 0; i < selfen_dyn_mastpart.size(); ++i) {
        so = selfen_dyn_mastpart.global2dIndex(i);
        selfen_dyn_mastpart[i].noalias() = Gimpmastpart[i].inverse() - Gbathmastpart[i].inverse() - m_selfen_static[so[0]];
        tmp = Gimpmastpart[i].inverse().cwiseAbs2();
        selfen_var_mastpart[i].noalias() = tmp * Gimpvarmastpart[i] * tmp;
    }
    
    const auto loc_corr = std::any_cast<bool>(parameters.at("local correlation"));
    if (loc_corr) {  // Set block off-diagonal parts of self-energy to zero if adopting local correlation approximation
        if (m_ptr2Gimp->nSites() % 2 != 0) throw std::invalid_argument("Number of sites must be even to consider local correlation approximation!");
        const std::size_t ns_2 = m_ptr2Gimp->nSites() / 2;
        for (std::size_t i = 0; i < selfen_dyn_mastpart.size(); ++i) {
            selfen_dyn_mastpart[i].bottomLeftCorner(ns_2, ns_2) = Eigen::MatrixXcd::Zero(ns_2, ns_2);
            selfen_dyn_mastpart[i].topRightCorner(ns_2, ns_2) = Eigen::MatrixXcd::Zero(ns_2, ns_2);
            selfen_var_mastpart[i].bottomLeftCorner(ns_2, ns_2) = Eigen::MatrixXd::Zero(ns_2, ns_2);
            selfen_var_mastpart[i].topRightCorner(ns_2, ns_2) = Eigen::MatrixXd::Zero(ns_2, ns_2);
        }
        for (int s = 0; s < 2; ++s) {
            m_selfen_static[s].bottomLeftCorner(ns_2, ns_2) = Eigen::MatrixXcd::Zero(ns_2, ns_2);
            m_selfen_static[s].topRightCorner(ns_2, ns_2) = Eigen::MatrixXcd::Zero(ns_2, ns_2);
            for (int n = 0; n < m_selfen_moms.dim1(); ++n) {
                m_selfen_moms(s, n).bottomLeftCorner(ns_2, ns_2) = Eigen::MatrixXcd::Zero(ns_2, ns_2);
                m_selfen_moms(s, n).topRightCorner(ns_2, ns_2) = Eigen::MatrixXcd::Zero(ns_2, ns_2);
            }
        }
    }
    selfen_dyn_mastpart.allGather();  // For analytic continuation after calling this method
    selfen_var_mastpart.allGather();
}

// Update the lattice Green's function using the current self-energy
void DMFTIterator::updateLatticeGF() {
    if ((m_ptr2H0->type() == "bethe" || m_ptr2H0->type() == "bethe_dimer") && m_iter > 0) m_Glat.mastFlatPart()() = m_ptr2Gimp->fourierCoeffs().mastFlatPart()();
    else computeLattGFfCoeffs(*m_ptr2H0, m_selfen_dyn, m_selfen_static, 1i * m_ptr2Gimp->matsubFreqs(), m_Glat);
    const auto loc_corr = std::any_cast<bool>(parameters.at("local correlation"));
    if (loc_corr) {  // Set block off-diagonal parts of lattice Green's function to zero if adopting local correlation approximation
        if (m_ptr2Gimp->nSites() % 2 != 0) throw std::invalid_argument("Number of sites must be even to consider local correlation approximation!");
        const std::size_t ns_2 = m_ptr2Gimp->nSites() / 2;
        auto Glatmastpart = m_Glat.mastFlatPart();
        for (std::size_t i = 0; i < Glatmastpart.size(); ++i) {
            Glatmastpart[i].bottomLeftCorner(ns_2, ns_2) = Eigen::MatrixXcd::Zero(ns_2, ns_2);
            Glatmastpart[i].topRightCorner(ns_2, ns_2) = Eigen::MatrixXcd::Zero(ns_2, ns_2);
        }
    }
}

std::pair<bool, double> DMFTIterator::checkConvergence() const {
    const auto convergtype = std::any_cast<std::string>(parameters.at("convergence type"));
    const auto prec = std::any_cast<double>(parameters.at("convergence criterion"));
    std::pair<bool, double> convergence(false, 0.0);
    
    if (convergtype == "G0_average_error") {
        convergence.second = (m_ptr2Gbath->valsOnTauGrid().mastFlatPart()() - m_G0old.mastFlatPart()()).squaredNorm();
        // Sum the accumulated squared norms on all processes to obtain the complete squared norm for Green's function difference
        MPI_Allreduce(MPI_IN_PLACE, &convergence.second, 1, MPI_DOUBLE, MPI_SUM, m_ptr2Gbath->valsOnTauGrid().mpiCommunicator());
        convergence.second = std::sqrt( convergence.second / (2 * m_ptr2Gbath->tauGridSize() * m_ptr2Gimp->nSites() * m_ptr2Gimp->nSites()) );
    }
    else if (convergtype == "G0_max_error") {
        convergence.second = (m_ptr2Gbath->valsOnTauGrid().mastFlatPart()() - m_G0old.mastFlatPart()()).cwiseAbs().maxCoeff();
        // Find the global maximum difference
        MPI_Allreduce(MPI_IN_PLACE, &convergence.second, 1, MPI_DOUBLE, MPI_MAX, m_ptr2Gbath->valsOnTauGrid().mpiCommunicator());
    }
    else if (convergtype == "Gimp_Glat_average_error") {
        convergence.second = (m_ptr2Gimp->fourierCoeffs().mastFlatPart()() - m_Glat.mastFlatPart()()).squaredNorm();
        // Sum the accumulated squared norms on all processes to obtain the complete squared norm for Green's function difference
        MPI_Allreduce(MPI_IN_PLACE, &convergence.second, 1, MPI_DOUBLE, MPI_SUM, m_ptr2Gimp->fourierCoeffs().mpiCommunicator());
        convergence.second = std::sqrt( convergence.second / (2 * (m_ptr2Gimp->freqCutoff() + 1) * m_ptr2Gimp->nSites() * m_ptr2Gimp->nSites()) );
    }
    else {  // Default to Gimp_Glat_max_error
        convergence.second = (m_ptr2Gimp->fourierCoeffs().mastFlatPart()() - m_Glat.mastFlatPart()()).cwiseAbs().maxCoeff();
        // Find the global maximum difference
        MPI_Allreduce(MPI_IN_PLACE, &convergence.second, 1, MPI_DOUBLE, MPI_MAX, m_ptr2Gimp->fourierCoeffs().mpiCommunicator());
    }
    
    if (convergence.second < prec) convergence.first = true;
    
    return convergence;
}

