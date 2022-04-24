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


DMFTIterator::DMFTIterator(std::shared_ptr<const BareHamiltonian> H0, std::shared_ptr<BareGreenFunction> Gbath, std::shared_ptr<const GreenFunction> Gimp) : m_ptr2H0(H0), m_ptr2Gbath(Gbath), m_ptr2Gimp(Gimp), m_Glat(2, Gimp->freqCutoff() + 1, Gimp->nSites(), Gimp->fourierCoeffs().mpiCommunicator()), m_selfen(2, Gimp->freqCutoff() + 1, Gimp->nSites(), Gimp->fourierCoeffs().mpiCommunicator()), m_iter(0) {
    // Default parameters
    parameters["G0 update step size"] = 1.0;
    parameters["convergence type"] = std::string("average_error");  // Or "max_error"
    parameters["convergence criterion"] = 0.005;
    // parameters["integration spline type"] = std::string("akima");
    
    // Compute high-frequency expansion coefficients of bath Green's function, which only depend on bare Hamiltonian
    // and thus only need to be calculated once when bare Hamiltonian passed in.
    m_ptr2Gbath->computeHighFreqExpan(*m_ptr2H0);
}

// Update the bath Green's function using the current lattice Green's function and self-energy
void DMFTIterator::updateBathGF() {
    ++m_iter;
    
    auto stepsize = std::any_cast<double>(parameters.at("G0 update step size"));
    
    if (stepsize < 0 || stepsize > 1) throw std::invalid_argument( "Step size for updating bath Green's function must be in [0, 1]!" );
    
    if (m_iter == 1) stepsize = 1.0;
    
    auto Gbathmastpart = m_ptr2Gbath->fourierCoeffs().mastFlatPart();
    auto Glatmastpart = m_Glat.mastFlatPart();
    
    if (m_ptr2H0->type() == "bethe") {
        if (m_ptr2Gimp->nSites() != 1) throw std::range_error("Number of sites must be 1 for Bethe lattice (with semicircular DOS)!");
        const std::complex<double> t = m_ptr2H0->hopMatElem(0);
        std::complex<double> iwu(m_ptr2H0->chemPot(), 0.0);
        std::array<std::size_t, 2> so;
        
        for (std::size_t i = 0; i < Glatmastpart.size(); ++i) {
            so = Glatmastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
            iwu.imag(m_ptr2Gimp->matsubFreqs()(so[1]));
            Gbathmastpart(i, 0, 0) *= 1.0 - stepsize;  // Number of sites is 1
            Gbathmastpart(i, 0, 0) += stepsize / (-iwu - (t * t) * Glatmastpart(i, 0, 0));  // Number of sites is 1
        }
    }
    else if (m_ptr2H0->type() == "bethe_dimer") {
        if (m_ptr2Gimp->nSites() != 2) throw std::range_error("Number of sites must be 2 for dimer Hubbard model with semicircular density of states!");
        const std::complex<double> t = m_ptr2H0->hopMatElem(0);
        const std::complex<double> tz = m_ptr2H0->hopMatElem(1);
        std::array<std::size_t, 2> so;
        
        Eigen::Matrix2cd zeta;
        
        zeta << m_ptr2H0->chemPot(), tz,
                tz,             m_ptr2H0->chemPot();
        
        for (std::size_t i = 0; i < Glatmastpart.size(); ++i) {
            so = Glatmastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
            zeta(0, 0).imag(m_ptr2Gimp->matsubFreqs()(so[1]));
            zeta(1, 1).imag(m_ptr2Gimp->matsubFreqs()(so[1]));
            Gbathmastpart[i] *= 1.0 - stepsize;
            Gbathmastpart[i].noalias() += stepsize * (-zeta - (t * t) * Glatmastpart[i]).inverse();
        }
    }
    else {
        auto selfemastpart = m_selfen.mastFlatPart();
        for (std::size_t i = 0; i < Glatmastpart.size(); ++i) {
            Gbathmastpart[i] *= 1.0 - stepsize;
            Gbathmastpart[i].noalias() += stepsize * (Glatmastpart[i].inverse() - selfemastpart[i]).inverse();
        }
    }
    
    // Do all-gather because every process needs full access to the Fourier coefficients of the Bath Green's function
    // for measurements in the next QMC run
    Gbathmastpart.allGather();
    // Fourier inversion does not require the Fourier coefficients have been all-gathered. And it already spreads the
    // complete inversion result to all processes for the next QMC run.
    m_ptr2Gbath->invFourierTrans();   // Spline built using already-calculated high-frequency expansion
}

// Approximate self-energy from the solved impurity problem
void DMFTIterator::approxSelfEnergy() {
    auto selfenmastpart = m_selfen.mastFlatPart();
    const auto Gimpmastpart = m_ptr2Gimp->fourierCoeffs().mastFlatPart();
    auto Gbathmastpart = m_ptr2Gbath->fourierCoeffs().mastFlatPart();
    for (std::size_t i = 0; i < selfenmastpart.size(); ++i) selfenmastpart[i].noalias() = Gimpmastpart[i].inverse() - Gbathmastpart[i].inverse();
}

// Update the lattice Green's function using the current self-energy
void DMFTIterator::updateLatticeGF() {
    if ((m_ptr2H0->type() == "bethe" || m_ptr2H0->type() == "bethe_dimer") && m_iter > 1) m_Glat.mastFlatPart()() = m_ptr2Gimp->fourierCoeffs().mastFlatPart()();
    else {
//        std::array<std::size_t, 2> so;
//        const std::size_t nc = _Gimp->nSites();
//        std::complex<double> iwu(_H0->mu, 0.0);
//
//        if (nc == 1) {  // Single site case, where we can utilize the noninteracting density of states
//            const std::size_t nbins = _H0->dos().size();
//            if (nbins == 0) throw std::range_error("DOS has not been computed or set!");
//            std::size_t ie;
//            const double binsize = (_H0->energyRange()[1] - _H0->energyRange()[0]) / nbins;
//            double e;
//            SqMatArray<std::complex<double>, 1, Eigen::Dynamic, 1> integrand(nbins, 1);  // Purly local to each process; include two end points where dos is zero
//
//            for (std::size_t i = 0; i < Glat.mastPartSize(); ++i) {
//                so = Glat.index2DinPart(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
//                iwu.imag((2 * so[1] + 1) * M_PI / _Gbath->inverseTemperature());
//                // Glat.masteredPart(i).setZero();
//                // Update the lattice Green's function
//                for (ie = 0; ie < nbins; ++ie) {
//                    e = (ie + 0.5) * binsize + _H0->energyRange()[0];
//                    integrand[ie](0) = -_H0->dos()(ie) / (iwu - e - selfenergy.masteredPart(i)(0));
//                    // Glat.masteredPart(i).noalias() -= binsize * _H0->dos()(ie) * ((1i * w + _H0->mu - e) * Eigen::MatrixXcd::Identity(nc, nc) - selfenergy.masteredPart(i)).inverse();
//                }
//                simpsonIntegrate(integrand, binsize, Glat.masteredPart(i));
//                // Add head and tail parts to the integral
//                Glat.masteredPart(i) += (binsize / 4) * (integrand[0] + integrand[nbins - 1]);
//            }
//        }
//        else if (nc > 1) {
//            const std::size_t spatodim = _H0->kPrimVecs().rows();
//            if (spatodim == 0) throw std::range_error("Reciprocal primative vectors have not been set!");
//            // else if (spatodim > 2) throw std::range_error("k-space integration has only been implemented for 1- and 2-dimensional cases!");
//            // SqMatArray<std::complex<double>, 1, Eigen::Dynamic, Eigen::Dynamic> integrand0(nk, nc);
//
//            if (_H0->type() == "dimer_mag_2d") {
//                if (_H0->hamDimerMag2d().size() == 0) throw std::range_error("Block Hamiltonian of the 2D dimer Hubbard model in magnetic fields has not been computed!");
//                std::size_t ist;
//                for (std::size_t i = 0; i < Glat.mastPartSize(); ++i) {
//                    so = Glat.index2DinPart(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
//                    iwu.imag((2 * so[1] + 1) * M_PI / _Gbath->inverseTemperature());
//                    Glat.masteredPart(i).setZero();
//                    for (ist = 0; ist < _H0->hamDimerMag2d().size(); ++ist) {
//                        Glat.masteredPart(i) += -(iwu * Eigen::Matrix2cd::Identity() - _H0->hamDimerMag2d()[ist] - selfenergy.masteredPart(i)).inverse();
//                    }
//                    Glat.masteredPart(i) /= static_cast<double>(_H0->hamDimerMag2d().size());
//                }
//            }
//            else {
//                if (_H0->kGridSizes().size() == 0) throw std::range_error("Grid size for the reciprocal space has not been set!");
//                const std::size_t nk = _H0->kGridSizes().prod();
//                std::size_t ik;
//                Eigen::VectorXd k;
//                Eigen::MatrixXcd H;
//                for (std::size_t i = 0; i < Glat.mastPartSize(); ++i) {
//                    so = Glat.index2DinPart(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
//                    iwu.imag((2 * so[1] + 1) * M_PI / _Gbath->inverseTemperature());
//                    Glat.masteredPart(i).setZero();
//                    for (ik = 0; ik < nk; ++ik) {
//                        _H0->kVecAtIndex(ik, k);
//                        _H0->constructHamiltonian(k, H);
//                        Glat.masteredPart(i) += -(iwu * Eigen::MatrixXcd::Identity(nc, nc) - H.selfadjointView<Eigen::Lower>() * Eigen::MatrixXcd::Identity(nc, nc) - selfenergy.masteredPart(i)).inverse();
//                    }
//                    Glat.masteredPart(i) /= static_cast<double>(nk);
//                }
//            }
//        }
        computeLattGFfCoeffs(*m_ptr2H0, m_selfen, 1i * m_ptr2Gimp->matsubFreqs(), m_Glat);
    }
}

std::pair<bool, double> DMFTIterator::checkConvergence() const {
    const auto convergtype = std::any_cast<std::string>(parameters.at("convergence type"));
    const auto prec = std::any_cast<double>(parameters.at("convergence criterion"));
    std::pair<bool, double> convergence(false, 0.0);
    
    if (convergtype == "average_error") {
        convergence.second = (m_ptr2Gimp->fourierCoeffs().mastFlatPart()() - m_Glat.mastFlatPart()()).squaredNorm();
        // Sum the accumulated squared norms on all processes to obtain the complete squared norm for Green's function difference
        MPI_Allreduce(MPI_IN_PLACE, &convergence.second, 1, MPI_DOUBLE, MPI_SUM, m_ptr2Gimp->fourierCoeffs().mpiCommunicator());
        convergence.second = std::sqrt( convergence.second / (2 * (m_ptr2Gimp->freqCutoff() + 1) * m_ptr2Gimp->nSites() * m_ptr2Gimp->nSites()) );
    }
    else if (convergtype == "max_error") {
        convergence.second = (m_ptr2Gimp->fourierCoeffs().mastFlatPart()() - m_Glat.mastFlatPart()()).cwiseAbs().maxCoeff();
        // Find the global maximum difference
        MPI_Allreduce(MPI_IN_PLACE, &convergence.second, 1, MPI_DOUBLE, MPI_MAX, m_ptr2Gimp->fourierCoeffs().mpiCommunicator());
    }
    
    if (convergence.second < prec) {
        convergence.first = true;
    }
    
    return convergence;
}

