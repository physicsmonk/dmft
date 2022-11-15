//
//  self_consistency.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef self_consistency_hpp
#define self_consistency_hpp

#include <memory>
#include <map>
#include <any>
#include "bare_hamiltonian.hpp"
#include "green_function.hpp"


class DMFTIterator {
private:
    // Pointers to external bare Hamiltonian and Green's functions. Note these Green's functions differ from the canonical definition by a minus sign
    std::shared_ptr<const BareHamiltonian> m_ptr2H0;
    std::shared_ptr<BareGreenFunction> m_ptr2Gbath;
    std::shared_ptr<const GreenFunction> m_ptr2Gimp;
    
protected:
    SqMatArray2XXcd m_G0old;  // A copy of old G0 in imaginary-time space
    SqMatArray2XXcd m_Glat;  // This is the lattice Green's function in Matsubara frequency space (differing from the canonical definition by a minus sign)
    SqMatArray2XXcd m_selfen;
    SqMatArray21Xcd m_selfenstatic;
    std::size_t m_iter;  // The number of iterations
    
public:
    std::map<std::string, std::any> parameters;
    
    DMFTIterator(std::shared_ptr<const BareHamiltonian> H0, std::shared_ptr<BareGreenFunction> Gbath, std::shared_ptr<const GreenFunction> Gimp);
    
    std::size_t numIterations() const {return m_iter;}
    void incrementIter() {++m_iter;}
    void resetIterator() {m_iter = 0;}
    
    void updateBathGF();
    
    void approxSelfEnergy(const bool loc_corr);
    
    void updateLatticeGF();
    
    std::pair<bool, double> checkConvergence() const;
    
    SqMatArray2XXcd& selfEnergy() {return m_selfen;}
    const SqMatArray2XXcd& selfEnergy() const {return m_selfen;}
    const auto selfEnHighFreq(const int spin, const std::complex<double> z) const {
        return m_ptr2Gbath->highFreqCoeffs()(spin, 0) - m_ptr2Gimp->highFreqCoeffs()(spin, 0)
        + (m_ptr2Gimp->highFreqCoeffs()(spin, 1) - m_ptr2Gimp->highFreqCoeffs()(spin, 0) * m_ptr2Gimp->highFreqCoeffs()(spin, 0)
           - m_ptr2Gbath->highFreqCoeffs()(spin, 1) + m_ptr2Gbath->highFreqCoeffs()(spin, 0) * m_ptr2Gbath->highFreqCoeffs()(spin, 0)) / z;
    }
    const SqMatArray21Xcd& selfEnStaticPart() const {return m_selfenstatic;}
};






// Calculate longitudinal conductivity
template <int n0, int n1, int nm>
double longitConduc(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen, const double beta, const double low, const double high, const double del) {
    Eigen::ArrayXd integrand = Eigen::ArrayXd::Zero(selfen.dim1());
    std::size_t iw, ik, m0, m1;
    const double dw = (high - low) / (selfen.dim1() - 1);
    const Eigen::ArrayXd ws = Eigen::ArrayXd::LinSpaced(selfen.dim1(), low, high);
    const Eigen::ArrayXd ebws = (beta * ws).exp();  // Save results of rather expensive exponential calculations
    int s;
    double sigmaxx = 0.0;
    
    if (H0.type() == "dimer_mag_2d") {
        const std::size_t nb_2 = H0.hamDimerMag2d().dim1();
        Eigen::Matrix2cd a;
        SqMatArray<std::complex<double>, 1, Eigen::Dynamic, 2> A(1, nb_2, 2);  // Purely local
        const auto Hmastpart = H0.hamDimerMag2d().mastDim0Part();
        
        for (s = 0; s < 2; ++s) {
            for (iw = 0; iw < selfen.dim1(); ++iw) {
                for (ik = 0; ik < H0.fermiVdimerMag2d().dim1(); ++ik) {  // dim1() is local size of the number of k-points
                    for (m0 = 0; m0 < nb_2; ++m0) {
                        a.noalias() = ((ws(iw) + 1i * del + H0.chemPot()) * Eigen::Matrix2cd::Identity() - Hmastpart(ik, m0) - selfen(s, iw)).inverse();
                        A[m0] = (a - a.adjoint()) / (2i * M_PI);
                        integrand(iw) -= A[m0].cwiseAbs2().sum() * std::norm(H0.fermiVdimerMag2d()(0, ik, m0, m0));
                    }
                    for (m1 = 0; m1 < nb_2 - 1; ++m1) {
                        for (m0 = m1 + 1; m0 < nb_2; ++m0)
                            integrand(iw) -= 2.0 * std::real(A[m0].cwiseProduct(A[m1].conjugate()).sum()) * std::norm(H0.fermiVdimerMag2d()(0, ik, m0, m1));
                    }
                }
            }
        }
        integrand *= -beta * ebws * (1.0 + ebws).square().inverse();
        sigmaxx = simpsonIntegrate(integrand, dw) / H0.hamDimerMag2d().size() * 2.0 * M_PI * M_PI;   // In units of e^2 / (2 * pi * hbar)
        MPI_Allreduce(MPI_IN_PLACE, &sigmaxx, 1, MPI_DOUBLE, MPI_SUM, selfen.mpiCommunicator());
    }
    
    return sigmaxx;
}

// Calculate Hall conductivity
template <int n0, int n1, int nm>
double hallConduc(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen, const double beta, const double low, const double high, const double del) {
    Eigen::ArrayXd integrand0(selfen.dim1()), integrand1 = Eigen::ArrayXd::Zero(selfen.dim1());
    std::size_t iw0, iw1, ik, m0, m1;
    const double dw = (high - low) / (selfen.dim1() - 1);
    const Eigen::ArrayXd ws = Eigen::ArrayXd::LinSpaced(selfen.dim1(), low, high);
    const Eigen::ArrayXd fws = 1.0 / (1.0 + (beta * ws).exp());
    int s;
    double sigmaxy = 0.0;
    const double nu = 1e-6;
    
    if (H0.type() == "dimer_mag_2d") {
        const std::size_t nb_2 = H0.hamDimerMag2d().dim1();
        Eigen::Matrix2cd a;
        SqMatArray<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, 2> A(selfen.dim1(), nb_2, 2);  // Purely local
        const auto Hmastpart = H0.hamDimerMag2d().mastDim0Part();
        
        for (s = 0; s < 2; ++s) {
            for (ik = 0; ik < H0.fermiVdimerMag2d().dim1(); ++ik) {  // dim1() is local size of the number of k-points
                for (iw0 = 0; iw0 < selfen.dim1(); ++iw0) {
                    for (m0 = 0; m0 < nb_2; ++m0) {
                        a.noalias() = ((ws(iw0) + 1i * del + H0.chemPot()) * Eigen::Matrix2cd::Identity() - Hmastpart(ik, m0) - selfen(s, iw0)).inverse();
                        A(iw0, m0) = (a - a.adjoint()) / (2i * M_PI);
                    }
                }
                for (iw1 = 0; iw1 < selfen.dim1(); ++iw1) {
                    integrand0.setZero();
                    for (iw0 = 0; iw0 < selfen.dim1(); ++iw0) {
                        for (m1 = 0; m1 < nb_2; ++m1) {
                            for (m0 = 0; m0 < nb_2; ++m0) {
                                integrand0(iw0) += std::imag(A(iw0, m0).cwiseProduct(A(iw1, m1).conjugate()).sum() * H0.fermiVdimerMag2d()(0, ik, m0, m1) * H0.fermiVdimerMag2d()(1, ik, m1, m0)) * (fws(iw0) - fws(iw1)) / ((ws(iw0) - ws(iw1) + nu) * (ws(iw0) - ws(iw1) + nu));
                            }
                        }
                    }
                    integrand1(iw1) += simpsonIntegrate(integrand0, dw);
                }
            }
        }
        sigmaxy = -simpsonIntegrate(integrand1, dw) / H0.hamDimerMag2d().size() * 2.0 * M_PI;   // In units of e^2 / (2 * pi * hbar)
        MPI_Allreduce(MPI_IN_PLACE, &sigmaxy, 1, MPI_DOUBLE, MPI_SUM, selfen.mpiCommunicator());
    }
    
    return sigmaxy;
}



#endif /* self_consistency_hpp */
