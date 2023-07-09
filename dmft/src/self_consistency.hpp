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
    //SqMatArray2XXcd m_G0old;  // A copy of old G0 in imaginary-time space
    SqMatArray2XXcd m_Glat;  // This is the lattice Green's function in Matsubara frequency space (differing from the canonical definition by a minus sign)
    SqMatArray2XXcd m_selfen_dyn;    // Dynamic part of self-energy
    SqMatArray21Xcd m_selfen_static;  // Static part of self-energy
    SqMatArray23Xcd m_selfen_moms;   // first to third moments (all Hermitian) of dynamic part of self-energy
    SqMatArray2XXd m_selfen_var;   // Self-energy variances
    Eigen::Index m_iter;  // The number of iterations
    
    void computeSelfEnMoms();
    
public:
    std::map<std::string, std::any> parameters;
    
    DMFTIterator(std::shared_ptr<const BareHamiltonian> H0, std::shared_ptr<BareGreenFunction> Gbath, std::shared_ptr<const GreenFunction> Gimp);
    
    Eigen::Index numIterations() const {return m_iter;}
    void incrementIter() {++m_iter;}
    void resetIterator() {m_iter = 0;}
    
    void updateBathGF();
    
    void approxSelfEnergy();
    
    void updateLatticeGF();
    
    std::pair<bool, double> checkConvergence() const;
    
    template <typename Derived, int n0, int n1, int nm, int n_mom>
    static void fitSelfEnMoms23(const Eigen::DenseBase<Derived>& matsfreqs, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen_dyn,
                                const SqMatArray<double, n0, n1, nm>& selfen_var, const Eigen::Index tailstart, SqMatArray<std::complex<double>, n0, n_mom, nm>& moms);
    
    SqMatArray2XXcd& dynSelfEnergy() {return m_selfen_dyn;}
    const SqMatArray2XXcd& dynSelfEnergy() const {return m_selfen_dyn;}
    SqMatArray21Xcd& staticSelfEnergy() {return m_selfen_static;}
    const SqMatArray21Xcd& staticSelfEnergy() const {return m_selfen_static;}
    SqMatArray23Xcd& selfEnergyMoms() {return m_selfen_moms;}
    const SqMatArray23Xcd& selfEnergyMoms() const {return m_selfen_moms;}
    SqMatArray2XXd& selfEnergyVar() {return m_selfen_var;}
    const SqMatArray2XXd& selfEnergyVar() const {return m_selfen_var;}
};




enum IntAlg : int {Trapezoidal = 0, CubicSpline = 1, Simpson = 2};

// Calculate longitudinal conductivity; Pass a intvec with any NaN to indicate equidistant energy grid and use Simpson integration
template <int n0, int n1, int nm, typename Derived, typename OtherDerived>
double longitConduc(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen, const double beta,
                    const Eigen::ArrayBase<Derived>& engrid, const Eigen::MatrixBase<OtherDerived>& intvec, const IntAlg intalg) {
    assert(selfen.dim1() == engrid.size());
    
    Eigen::ArrayXd integrand = Eigen::ArrayXd::Zero(selfen.dim1());
    Eigen::Index iw, ik, m0, m1;
    //const double dw = (high - low) / (selfen.dim1() - 1);
    //const Eigen::ArrayXd ws = Eigen::ArrayXd::LinSpaced(selfen.dim1(), low, high);
    const Eigen::ArrayXd ebws = (beta * engrid.real()).exp();  // Save results of rather expensive exponential calculations
    double sigmaxx = 0.0;
    
    if (H0.type() == "dimer_mag_2d") {
        const Eigen::Index nb_2 = H0.hamDimerMag2d().dim1();
        Eigen::Matrix2cd a;
        SqMatArray<std::complex<double>, 1, Eigen::Dynamic, 2> A(1, nb_2, 2);  // Purely local
        const auto Hmastpart = H0.hamDimerMag2d().mastDim0Part();
        
        for (int s = 0; s < 2; ++s) {
            for (iw = 0; iw < selfen.dim1(); ++iw) {
                for (ik = 0; ik < H0.fermiVdimerMag2d().dim1(); ++ik) {  // dim1() is local size of the number of k-points
                    for (m0 = 0; m0 < nb_2; ++m0) {
                        a.noalias() = ((engrid(iw) + H0.chemPot()) * Eigen::Matrix2cd::Identity() - Hmastpart(ik, m0) - selfen(s, iw)).inverse();
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
        // if (H0.hamDimerMag2d().procRank() == 0) std::cout << "integrand\n" << integrand << "\nend integrand" << std::endl;  // For testing
        integrand *= -beta * ebws / (1.0 + ebws).square();
        // In units of e^2 / (2 * pi * hbar)
        if (intalg == Simpson) sigmaxx = simpsonIntegrate(integrand, std::real(engrid(1)) - std::real(engrid(0))) / H0.hamDimerMag2d().size() * 2.0 * M_PI * M_PI;
        else if (intalg == CubicSpline) sigmaxx = intvec.dot(integrand.matrix()) / H0.hamDimerMag2d().size() * 2.0 * M_PI * M_PI;
        // Use trapezoidal integration
        else sigmaxx = ((integrand(Eigen::seq(1, Eigen::last)) + integrand(Eigen::seq(0, Eigen::last - 1))) / 2.0
            * (engrid(Eigen::seq(1, Eigen::last)).real() - engrid(Eigen::seq(0, Eigen::last - 1)).real())).sum() / H0.hamDimerMag2d().size() * 2.0 * M_PI * M_PI;
        MPI_Allreduce(MPI_IN_PLACE, &sigmaxx, 1, MPI_DOUBLE, MPI_SUM, selfen.mpiComm());
    }
    else throw std::range_error("Have not implemented calculation of longitudinal conductivity for H0.type() not being dimer_mag_2d");
    
    return sigmaxx;
}

// Calculate Hall conductivity; Pass a intvec with any NaN to indicate equidistant energy grid and use Simpson integration
template <int n0, int n1, int nm, typename Derived, typename OtherDerived>
double hallConduc(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen, const double beta,
                  const Eigen::ArrayBase<Derived>& engrid, const Eigen::MatrixBase<OtherDerived>& intvec, const IntAlg intalg) {
    assert(selfen.dim1() == engrid.size());
    
    Eigen::ArrayXd integrand0(selfen.dim1()), integrand1 = Eigen::ArrayXd::Zero(selfen.dim1());
    Eigen::Index iw0, iw1, ik, m0, m1;
    //const double dw = (high - low) / (selfen.dim1() - 1);
    //const Eigen::ArrayXd ws = Eigen::ArrayXd::LinSpaced(selfen.dim1(), low, high);
    const Eigen::ArrayXd fws = 1.0 / (1.0 + (beta * engrid.real()).exp());
    double endiff, fermifac, sigmaxy = 0.0;
    const double nu = 1e-6;
    
    if (H0.type() == "dimer_mag_2d") {
        const Eigen::Index nb_2 = H0.hamDimerMag2d().dim1();
        Eigen::Matrix2cd a;
        SqMatArray<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, 2> A(selfen.dim1(), nb_2, 2);  // Purely local
        const auto Hmastpart = H0.hamDimerMag2d().mastDim0Part();
        
        for (int s = 0; s < 2; ++s) {
            for (ik = 0; ik < H0.fermiVdimerMag2d().dim1(); ++ik) {  // dim1() is local size of the number of k-points
                for (iw0 = 0; iw0 < selfen.dim1(); ++iw0) {
                    for (m0 = 0; m0 < nb_2; ++m0) {
                        a.noalias() = ((engrid(iw0) + H0.chemPot()) * Eigen::Matrix2cd::Identity() - Hmastpart(ik, m0) - selfen(s, iw0)).inverse();
                        A(iw0, m0) = (a - a.adjoint()) / (2i * M_PI);
                    }
                }
                for (iw1 = 0; iw1 < selfen.dim1(); ++iw1) {
                    integrand0.setZero();
                    for (iw0 = 0; iw0 < selfen.dim1(); ++iw0) {
                        endiff = std::real(engrid(iw0)) - std::real(engrid(iw1)) + nu;
                        fermifac = (fws(iw0) - fws(iw1)) / (endiff * endiff);
                        for (m1 = 0; m1 < nb_2; ++m1) {
                            for (m0 = 0; m0 < nb_2; ++m0) {
                                integrand0(iw0) += std::imag(A(iw0, m0).cwiseProduct(A(iw1, m1).conjugate()).sum() * H0.fermiVdimerMag2d()(0, ik, m0, m1)
                                                             * H0.fermiVdimerMag2d()(1, ik, m1, m0)) * fermifac;
                            }
                        }
                    }
                    if (intalg == Simpson) integrand1(iw1) += simpsonIntegrate(integrand0, std::real(engrid(1)) - std::real(engrid(0)));
                    else if (intalg == CubicSpline) integrand1(iw1) += intvec.dot(integrand0.matrix());
                    // Use trapezoidal integration
                    else integrand1(iw1) += ((integrand0(Eigen::seq(1, Eigen::last)) + integrand0(Eigen::seq(0, Eigen::last - 1))) / 2.0
                        * (engrid(Eigen::seq(1, Eigen::last)).real() - engrid(Eigen::seq(0, Eigen::last - 1)).real())).sum();
                }
            }
        }
        // In units of e^2 / (2 * pi * hbar)
        if (intalg == Simpson) sigmaxy = -simpsonIntegrate(integrand1, std::real(engrid(1)) - std::real(engrid(0))) / H0.hamDimerMag2d().size() * 2.0 * M_PI;
        else if (intalg == CubicSpline) sigmaxy = -intvec.dot(integrand1.matrix()) / H0.hamDimerMag2d().size() * 2.0 * M_PI;
        // Use trapezoidal integration
        else sigmaxy = -((integrand1(Eigen::seq(1, Eigen::last)) + integrand1(Eigen::seq(0, Eigen::last - 1))) / 2.0
            * (engrid(Eigen::seq(1, Eigen::last)).real() - engrid(Eigen::seq(0, Eigen::last - 1)).real())).sum() / H0.hamDimerMag2d().size() * 2.0 * M_PI;
        MPI_Allreduce(MPI_IN_PLACE, &sigmaxy, 1, MPI_DOUBLE, MPI_SUM, selfen.mpiComm());
    }
    else throw std::range_error("Have not implemented calculation of longitudinal conductivity for H0.type() not being dimer_mag_2d");
    
    return sigmaxy;
}

// Hall conductivity over p / q in the zero field limit
template <int n0, int n1, int nm, typename Derived, typename OtherDerived>
double hallConducCoeff(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen, const double beta,
                       const Eigen::ArrayBase<Derived>& engrid, const Eigen::MatrixBase<OtherDerived>& intvec, const IntAlg intalg) {
    assert(selfen.dim1() == engrid.size());
    
    Eigen::ArrayXd integrand = Eigen::ArrayXd::Zero(selfen.dim1());
    //const double dw = (high - low) / (selfen.dim1() - 1);
    //const Eigen::ArrayXd ws = Eigen::ArrayXd::LinSpaced(selfen.dim1(), low, high);
    const Eigen::ArrayXd ebws = (beta * engrid.real()).exp();  // Save results of rather expensive exponential calculations
    double sigmaHxy = 0.0;
    
    if (H0.type() == "dimer_mag_2d") {
        if(H0.hamDimerMag2d().dim1() > 1) throw std::runtime_error("hallConducCoeff: Magnetic field is not set to zero");
        Eigen::Matrix2cd A;
        Eigen::VectorXd k;
        Eigen::MatrixXcd epsyy;
        const auto Hmastpart = H0.hamDimerMag2d().mastDim0Part();
        
        for (int s = 0; s < 2; ++s) {
            for (Eigen::Index iw = 0; iw < selfen.dim1(); ++iw) {
                for (Eigen::Index kidlocal = 0; kidlocal < H0.fermiVdimerMag2d().dim1(); ++kidlocal) {  // dim1() is local size of the number of k-points
                    A.noalias() = ((engrid(iw) + H0.chemPot()) * Eigen::Matrix2cd::Identity() - Hmastpart(kidlocal, 0) - selfen(s, iw)).inverse();
                    A = (A - A.adjoint().eval()) / (2i * M_PI);
                    H0.kVecAtFlatId(kidlocal + Hmastpart.displ(), k);
                    H0.constructBandCurvature(1, 1, k, epsyy);
                    integrand(iw) += H0.fermiVdimerMag2d()(0, kidlocal, 0, 0).real() * H0.fermiVdimerMag2d()(0, kidlocal, 0, 0).real() * epsyy(0, 0).real()
                                     * (A * A * A).trace().real();
                }
            }
        }
        // if (H0.hamDimerMag2d().procRank() == 0) std::cout << "integrand\n" << integrand << "\nend integrand" << std::endl;  // For testing
        integrand *= -beta * ebws / (1.0 + ebws).square();
        // In units of e^2 / (2 * pi * hbar)
        if (intalg == Simpson) sigmaHxy = simpsonIntegrate(integrand, std::real(engrid(1)) - std::real(engrid(0))) / H0.hamDimerMag2d().size()
            * 8.0 * M_PI * M_PI * M_PI * M_PI / 3.0;
        else if (intalg == CubicSpline) sigmaHxy = intvec.dot(integrand.matrix()) / H0.hamDimerMag2d().size() * 8.0 * M_PI * M_PI * M_PI * M_PI / 3.0;
        // Use trapezoidal integration
        else sigmaHxy = ((integrand(Eigen::seq(1, Eigen::last)) + integrand(Eigen::seq(0, Eigen::last - 1))) / 2.0
            * (engrid(Eigen::seq(1, Eigen::last)).real() - engrid(Eigen::seq(0, Eigen::last - 1)).real())).sum() / H0.hamDimerMag2d().size()
            * 8.0 * M_PI * M_PI * M_PI * M_PI / 3.0;
        MPI_Allreduce(MPI_IN_PLACE, &sigmaHxy, 1, MPI_DOUBLE, MPI_SUM, selfen.mpiComm());
    }
    else throw std::range_error("Have not implemented calculation of longitudinal conductivity for H0.type() not being dimer_mag_2d");
    
    return sigmaHxy;
}


#endif /* self_consistency_hpp */
