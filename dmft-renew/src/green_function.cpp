//
//  green_function.cpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//
// Bare Green's function (Weiss field) in this program is related to the conventional bare Green's function by
// G0^-1(iw) = -(G0conv^-1(iw) - U / 2), i.e., the chemical potential is shifted by -U / 2 and G0(tau) > 0 for
// 0 <= tau <= beta.

#include <iostream>
#include <cmath>
#include "green_function.hpp"



GenericGreenFunction::GenericGreenFunction(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const MPI_Comm& comm) :
m_beta(beta),
m_matsfs(Eigen::ArrayXd::LinSpaced(nfcut + 1, M_PI / beta, (2 * nfcut + 1) * M_PI / beta)),
m_imagts(Eigen::ArrayXd::LinSpaced(ntau, 0.0, beta)),
m_Gw(2, nfcut + 1, nc, comm), m_Gt(2, ntau, nc, comm), m_Ghfc(2, 2, nc) {
    if (ntau < 2) throw std::invalid_argument( "Tau grid size of GenericGreenFunction cannot be less than 2!" );
}

// Gw will not be all-gathered
void GenericGreenFunction::fourierTransform() {
    auto Gwmastpart = m_Gw.mastFlatPart();
    std::array<std::size_t, 2> so;
    
    m_Gspl.build(&m_imagts, &m_Gt, m_Ghfc);
    
    for (std::size_t i = 0; i < Gwmastpart.size(); ++i) {
        so = Gwmastpart.global2dIndex(i);
        m_Gspl.fourierTransform(so[0], m_matsfs(so[1]), Gwmastpart[i]);
    }
}

void GenericGreenFunction::invFourierTrans() {
// Fourier coefficients Gw are imaginarily partitioned into all processes
    if (!isDiscretized()) throw std::domain_error("Cannot inverse Fourier transform GenericGreenFunction because tau is not discretized (tau grid size > 1)!");
    
    auto Gwmastpart = m_Gw.mastFlatPart();
    std::array<std::size_t, 2> so;
    std::size_t i, t;
    int s;
    std::complex<double> ff;
    Eigen::MatrixXcd gw_no_high_freq(nSites(), nSites());
    
    m_Gt().setZero();
    for (i = 0; i < Gwmastpart.size(); ++i) {
        so = Gwmastpart.global2dIndex(i);  // Get index in (spin, omega) space w.r.t. the full-sized data
        gw_no_high_freq = Gwmastpart[i] - fCoeffHighFreq(static_cast<int>(so[0]), 1i * m_matsfs(so[1]));
        for (t = 0; t < tauGridSize() - 1; ++t) {
            ff = std::exp(1i * std::fmod(m_matsfs(so[1]) * m_imagts(t), 2 * M_PI));
            // G(tau) on each process accumulates the part of the Fourier coefficients mastered by that process
            m_Gt(so[0], t) += (gw_no_high_freq / ff + gw_no_high_freq.adjoint() * ff) / m_beta;
        }
    }
    // Now sum the partially summed results on every process, to obtain the complete Fourier inversion
    m_Gt.allSum();
    // Add the high frequency expansion part
    for (s = 0; s < 2; ++s) {for (t = 0; t < tauGridSize() - 1; ++t) m_Gt(s, t) += valAtTauHighFreq(s, m_imagts(t));}
    
    for (int s = 0; s < 2; ++s) {
//        // Special treatment for tau = 0, accounting for discontinuity of Green's function at tau = 0: Gij(0+) - Gij(0-) = delta_ij.
//        // What is computed is Gij(0), which equals (Gij(0+) + Gij(0-)) / 2, but what should be stored is Gij(0+), which is just
//        // Gij(0) + delta_ij / 2.
//        G(s, 0) += Eigen::MatrixXcd::Identity(nSites(), nSites()) / 2.0;
        
        // Fill Gij(beta-) using relation Gij(0+) + Gij(beta-) = delta_ij
        m_Gt(s, tauGridSize() - 1) = Eigen::MatrixXcd::Identity(nSites(), nSites()) - m_Gt(s, 0);
    }
    //}
}

// Get Green's function value at extended (discretized) tau: -beta < tau < beta.
// tau = beta in program means tau = beta- in physics, etc.
// Note that we only stored G(tau) for 0 < tau < beta, so other values are retrieved by Green's function's symmetries.
// Optional "approach = 1" defines the direction for tau to approach zero, and is only used when tau = 0.
std::complex<double> GenericGreenFunction::valAtExtendedTau(const int spin, const std::size_t x1, const std::size_t x2, int ext_tau_ind, const LimitDirection approach) const {
    // Cast to int because ext_tau_ind can be negative
    assert( tauGridSize() > 0 && ext_tau_ind > -static_cast<int>(tauGridSize()) && ext_tau_ind < static_cast<int>(tauGridSize()) );
    
    double sgn = 1.0;
    
    if (ext_tau_ind < 0 || (ext_tau_ind == 0 && approach == LeftLimit)) {  // Case of tau < 0: using antiperiodicity of Green's functions
        sgn = -1.0;
        ext_tau_ind += tauGridSize() - 1;
    }
    
    return sgn * m_Gt(spin, ext_tau_ind, x1, x2);
}

// Overloaded version for getting Green's function matrix in site space
const auto GenericGreenFunction::valAtExtendedTau(const int spin, int ext_tau_ind, const LimitDirection approach) const {
    // Cast to int because ext_tau_ind can be negative
    assert( tauGridSize() > 0 && ext_tau_ind > -static_cast<int>(tauGridSize()) && ext_tau_ind < static_cast<int>(tauGridSize()) );
    
    double sgn = 1.0;
    
    if (ext_tau_ind < 0 || (ext_tau_ind == 0 && approach == LeftLimit)) {  // Case of tau < 0: using antiperiodicity of Green's functions
        sgn = -1.0;
        ext_tau_ind += tauGridSize() - 1;
    }
    
    return sgn * m_Gt(spin, ext_tau_ind);
}

std::complex<double> GenericGreenFunction::interpValAtExtendedTau(const std::size_t spin, const std::size_t x1, const std::size_t x2, double tau) const {
    assert(tau >= -m_beta && tau <= m_beta);
    
    // Green's function has a period of 2 * beta; now -beta <= rtau <= beta
    // const double rtau = tau - round(tau / (2 * beta)) * 2 * beta;
    // This round function rounds half-way up, so will swap the cases of tau = beta, -beta, leading to error w.r.t.
    // the discontinuity at tau = beta, -beta. This is because we considered tau = beta means tau = beta- and tau = -beta
    // means -beta+0. Also rounding number can get tricky. Anyway in program tau's range is always within [-beta, beta],
    // so we just save the effort for making this more general.
    
    double sgn = 1.0;
    
    if (tau < 0) {
        sgn = -1.0;
        tau += m_beta;
    }
    
    return sgn * m_Gspl.equidistAt(spin, tau, x1, x2);
}

// Eigen::Ref<Eigen::MatrixXcd> also references fixed size matrix
void GenericGreenFunction::interpValAtExtendedTau(const std::size_t spin, double tau, Eigen::Ref<Eigen::MatrixXcd> result) const {
    assert(tau >= -m_beta && tau <= m_beta);
    
    // Green's function has a period of 2 * beta; now -beta <= rtau <= beta
    // const double rtau = tau - round(tau / (2 * beta)) * 2 * beta;
    double sgn = 1.0;
    
    if (tau < 0) {
        sgn = -1.0;
        tau += m_beta;
    }
    
    std::size_t x0, x1;
    result.resize(nSites(), nSites());
    for (x1 = 0; x1 < nSites(); ++x1) {for (x0 = x1; x0 < nSites(); ++x0) result(x0, x1) = sgn * m_Gspl.equidistAt(spin, tau, x0, x1);}
    //result = result.template selfadjointView<Eigen::Lower>();
    result.triangularView<Eigen::Upper>() = result.adjoint();  // No aliasing issue
}

void GenericGreenFunction::setParams(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau) {
    if (ntau < 2) throw std::invalid_argument( "Tau grid size of GenericGreenFunction cannot be less than 2!" );
    
    m_beta = beta;
    m_matsfs = Eigen::ArrayXd::LinSpaced(nfcut + 1, M_PI / beta, (2 * nfcut + 1) * M_PI / beta);
    m_imagts = Eigen::ArrayXd::LinSpaced(ntau, 0.0, beta);
    m_Gw.resize(2, nfcut + 1, nc);  // No-op if sizes match; adapts to both MPI and non-MPI versions
    m_Gt.resize(2, ntau, nc);
    m_Ghfc.resize(2, 2, nc);
}







// First index counts tau (from 0 to beta) and second index counts omega
void BareGreenFunction::constructExpiwt() {
    if (m_eiwt.rows() < 2) throw std::invalid_argument("To construct expiwt array, tau grid size cannot be less than 2!");
    
    std::size_t t, o;
    const double dtau = m_beta / (m_eiwt.rows() - 1);
    
    for (o = 0; o < m_eiwt.cols(); ++o) {for (t = 0; t < m_eiwt.rows(); ++t) m_eiwt(t, o) = std::exp(1i * std::fmod(m_matsfs(o) * (t * dtau), 2 * M_PI));}
}

BareGreenFunction::BareGreenFunction(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const std::size_t nt4eiwt, const MPI_Comm& comm) : GenericGreenFunction(beta, nc, nfcut, ntau, comm) {
    if (nt4eiwt > 1) {
        m_eiwt.resize(nt4eiwt, nfcut + 1);
        constructExpiwt();
    }
//    else {  // nt4eiwt = 0, 1, in these cases construct expiwt array only for lowest positive frequency and a sufficiently
//        // large tau grid size for use in measuring S during QMC run
//        eiwt.resize(50001, 1);
//        constructExpiwt();
//    }
}

/* A little bit confusion might arise here. To clarify, the interacting Hamiltonian (including the chemical potential term) is written as
 
                                 K = H0 - mu * N + HU = H0 - (mu - U / 2) * N + (HU - U * N / 2).
 
   The last expression is used in the impurity solver to solve for the interacting Green's function, during which the partition function is
   expanded in terms of HU - U * N / 2 and the Weiss field is derived from H0 - (mu - U / 2) * N. thus the interacting part of the Hamiltonian
   is formally HU - U * N / 2 and the noninteracting part is formally K0 = H0 - (mu - U / 2) * N. Therefore, the bare Green's function (or the
   Weiss field) does not simply correspond to K with U switched to zero, but to K0, i.e., the chemical potential of the bare Green's function is
   the effective chemical potential mu - U / 2 used in the impurity solver. On the other hand, the chemical potential of the interacting Green's
   function that will be used in the high-frequency expansion is just the true chemical potential mu. */
void BareGreenFunction::computeHighFreqCoeffs(const BareHamiltonian& H0) {
    // H0.mu is the effective chemical potential
    for (int s = 0; s < 2; ++s) {
        m_Ghfc(s, 0) = -(H0.moments()(s, 0) - H0.chemPot() * Eigen::MatrixXcd::Identity(nSites(), nSites()));
        m_Ghfc(s, 1) = H0.moments()(s, 1) - 2.0 * H0.chemPot() * H0.moments()(s, 0) + H0.chemPot() * H0.chemPot() * Eigen::MatrixXcd::Identity(nSites(), nSites());
    }
}

// Do Fourier inversion into discretized tau grid and create cubic spline with correct boundary conditions
void BareGreenFunction::invFourierTrans() {
    GenericGreenFunction::invFourierTrans();
    m_Gspl.build(&m_imagts, &m_Gt, m_Ghfc);
}

void BareGreenFunction::setParams(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const std::size_t nt4eiwt) {
    // Make a copy for old beta for helping determine whether needs to reconstruct expiwt array
    double oldbeta = m_beta;
    
    // Update member variables inherited from GenericGreenFunction
    GenericGreenFunction::setParams(beta, nc, nfcut, ntau);  // beta has been updated to beta_
    
    // Determine whether needs to reconstruct expiwt array
    if (tauGridSizeOfExpiwt() != nt4eiwt || m_eiwt.cols() != nfcut + 1 || std::fabs(oldbeta - beta) > 1e-9) {
        m_eiwt.resize(nt4eiwt, nfcut + 1);  // No-op if shapes match
        if (nt4eiwt > 1) constructExpiwt();
        else if (nt4eiwt == 1) std::cout << "Warning: specified tau grid size of expiwt array is 1 (less than 2) so will not construct expiwt array nor use it!" << std::endl;
    }
}






// Sw0 and Sw0var use MPI only to sum measurements on all processes
GreenFunction::GreenFunction(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau,
                             const std::size_t nbins4S, const MPI_Comm& comm) : GenericGreenFunction(beta, nc, nfcut, ntau, comm),
                                                          //Gwvar(2, nfcut + 1, nc, comm_),
                                                          m_S(2, nbins4S, nc, comm), m_dens(nc, 2), m_densvar(nc, 2) { }

void GreenFunction::computeHighFreqCoeffs(const BareHamiltonian& H0, const double U) {
    const auto I = Eigen::MatrixXd::Identity(nSites(), nSites());  // Identity matrix expression
    const double mu = H0.chemPot() + U / 2;   // This is the true chemical potential; H0.mu is the effective chemical potential.
    for (int s = 0; s < 2; ++s) {
        m_Ghfc(s, 0) = -(H0.moments()(s, 0) - mu * I + U * m_dens.col(1 - s).asDiagonal() * I);
        m_Ghfc(s, 1) = H0.moments()(s, 1) - 2.0 * mu * H0.moments()(s, 0) + mu * mu * I
            + U * m_dens.col(1 - s).asDiagonal() * (H0.moments()(s, 0) - mu * I) + U * (H0.moments()(s, 0) - mu * I) * m_dens.col(1 - s).asDiagonal()
            + U * U * m_dens.col(1 - s).asDiagonal() * I;
    }
}

// Each process only evaluate G on its mastered imaginary partition. No need to gather the Fourier coefficients because from here the
// flow is entering the DMFT equations. Return an estimation of Simpson integration error.
double GreenFunction::evalFromSelfEgf(const BareGreenFunction& G0) {
    if (nTauBins4selfEgf() == 0) throw std::domain_error( "Number of bins for S is zero so cannot evaluate Green's function from S!" );
    if (!isDiscretized()) throw std::domain_error( "GreenFunction in tau space was not discretized so cannot evaluate it from S!" );
    if (tauGridSize() != G0.tauGridSize()) throw std::range_error("Tau grid sizes of G and G0 do not match!");
    if (freqCutoff() != G0.freqCutoff()) throw std::range_error("Frequency cutoffs of G and G0 do not match!");
    
    auto Gmastpart = m_Gt.mastFlatPart();
    const auto G0mastpart = G0.valsOnTauGrid().mastFlatPart();
    const double binsize4S = m_beta / nTauBins4selfEgf();
    std::array<std::size_t, 2> i2d;
    std::size_t i, ibin4S;
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, Eigen::Dynamic> integrand(1, nTauBins4selfEgf(), nSites());  // Purely local to each process
    
    // Evaluate G in tau space.
    for (i = 0; i < Gmastpart.size(); ++i) {
        // Each process only evaluates the G's on its mastered partition
        i2d = Gmastpart.global2dIndex(i);
        // G0.interpValAtExtendedTau(i2d[0], tau, G.masteredPart(i));
        for (ibin4S = 0; ibin4S < nTauBins4selfEgf(); ++ibin4S) {
            G0.interpValAtExtendedTau(i2d[0], m_imagts(i2d[1]) - (ibin4S + 0.5) * binsize4S, integrand[ibin4S]);
            integrand[ibin4S] *= m_S(i2d[0], ibin4S);  // Automatically introduced a temporary here to deal with the aliasing issue
            // G.masteredPart(i).noalias() += g0 * S(i2d[0], ibin4S);   // This requires S is not scaled by its bin size
        }
        simpsonIntegrate<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>(integrand, binsize4S, Gmastpart[i]);  // Must explicitly instantiate
        // Also add head and tail parts to the integral
        Gmastpart[i] += (binsize4S / 4) * (integrand[0] + integrand[nTauBins4selfEgf() - 1]) + G0mastpart[i];
    }
    Gmastpart.allGather();  // All-gather for next Fourier transform
    
    // Calculate Simpson integration error
    double interr = 0.0, interr1;
    for (int s = 0; s < 2; ++s) {
        // Correct Gij(beta-) using relation Gij(0+) + Gij(beta-) = delta_ij
        m_Gt(s, tauGridSize() - 1) = Eigen::MatrixXcd::Identity(nSites(), nSites()) - m_Gt(s, 0);
        
        interr1 = (m_Gt(s, tauGridSize() - 1).diagonal().real() - m_dens.col(s)).cwiseAbs().maxCoeff();
        if (interr1 > interr) interr = interr1;
    }
    
    fourierTransform();   // Only evaluate to mastered partition
    
    return interr;
}

void GreenFunction::setParams(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const std::size_t nbins4S) {
    GenericGreenFunction::setParams(beta, nc, nfcut, ntau);   // Set member variables inherited from GenericGreenFunction
    
    //Gwvar.resize(2, nfcut + 1, nc);  // No-op if sizes match
    m_S.resize(2, nbins4S, nc);
    m_dens.resize(nc, 2);
    m_densvar.resize(nc, 2);
}



