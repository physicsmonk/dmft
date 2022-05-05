//
//  green_function.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef green_function_hpp
#define green_function_hpp

#include <complex>
#include "cubic_spline.hpp"
#include "bare_hamiltonian.hpp"


using namespace std::complex_literals;

enum LimitDirection : int {LeftLimit = -1, RightLimit = 1};


class GenericGreenFunction {
protected:
    double m_beta;
    Eigen::ArrayXd m_matsfs, m_imagts;  // Choose to explicitly store the Matsubara frequencies and imaginary times to minimize runtime recalculations
    // Gw only stores matrices at positive frequencies because of the relation conj(Gij(iw)) = Gji(-iw). Due to this relation,
    // G is Hermitian in site space so essentially we can just store its upper triangular part, nontheless we store the
    // whole matrix because this is natural when calculated by Fourier inversion and this can also facilitate some calculations.
    SqMatArray2XXcd m_Gw, m_Gt;
    // Coefficient matrices of first- and second-order high-frequency expansion of Green's function:
    // G(iw) = -I / iw + G1 / (iw)^2 - G2 / (iw)^3 + ...
    // G1 = G'(0+) - G'(0-) = G'(0+) + G'(beta-), G2 = G"(0+) - G"(0-) = G"(0+) + G"(beta-).
    SqMatArray22Xcd m_Ghfc;
    CubicSplineMat2XXcd m_Gspl;  // Cubic spline for G(tau)
    
public:
    GenericGreenFunction() : m_beta(0.0) {}
    GenericGreenFunction(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const MPI_Comm& comm = MPI_COMM_SELF);
    GenericGreenFunction(const GenericGreenFunction&) = default;
    GenericGreenFunction(GenericGreenFunction&&) = default;
    GenericGreenFunction& operator=(const GenericGreenFunction&) = default;
    GenericGreenFunction& operator=(GenericGreenFunction&&) = default;
    
    virtual ~GenericGreenFunction() {}
    
    double inverseTemperature() const {return m_beta;}
    
    std::size_t tauGridSize() const {return m_Gt.dim1();}
    
    bool isDiscretized() const {return tauGridSize() > 1;}
    
    std::size_t freqCutoff() const {return m_Gw.dim1() - 1;}
    
    std::size_t nSites() const {return m_Gw.dimm();}
    
    const Eigen::ArrayXd& matsubFreqs() const {return m_matsfs;}
    const Eigen::ArrayXd& imagTimes() const {return m_imagts;}
    
    const SqMatArray2XXcd &fourierCoeffs() const {return m_Gw;}
    SqMatArray2XXcd &fourierCoeffs() {return m_Gw;}
    
    const SqMatArray2XXcd &valsOnTauGrid() const {return m_Gt;}
    SqMatArray2XXcd &valsOnTauGrid() {return m_Gt;}
    
    const SqMatArray22Xcd& highFreqCoeffs() const {return m_Ghfc;}
    // Return a const Eigen cwise-operation expression, whose type is too complicated to write
    const auto fCoeffHighFreq(const int spin, const std::complex<double> z) const {
        // We can safely return this Eigen expression because G1 and G2 are members of this class, not temporary objects within this function.
        // The first-order coefficient is -(G(0+) + G(beta-)) = -I.
        const std::complex<double> zsq = z * z;
        return -Eigen::MatrixXcd::Identity(nSites(), nSites()) / z + m_Ghfc(spin, 0) / zsq - m_Ghfc(spin, 1) / (zsq * z);
    }
    // Return a const Eigen cwise-operation expression, whose type is too complicated to write
    const auto valAtTauHighFreq(const int spin, const double tau) const {
        assert(tau >=0 && tau <= m_beta);
        // We can safely return this Eigen expression because G1 and G2 are members of this class, not temporary objects within this function
        return 0.5 * Eigen::MatrixXcd::Identity(nSites(), nSites()) + ((2.0 * tau - m_beta) / 4.0) * m_Ghfc(spin, 0) - ((m_beta - tau) * tau / 4.0) * m_Ghfc(spin, 1);
    }
    
    void fourierTransform();
    virtual void invFourierTrans();
    
    std::complex<double> valAtExtendedTau(const int spin, const std::size_t x1, const std::size_t x2, int ext_tau_ind, const LimitDirection approach) const;
    
    // Return a const Eigen cwise-operation expression, whose type is too complicated to write
    const auto valAtExtendedTau(const int spin, int ext_tau_ind, const LimitDirection approach) const;
    
    std::complex<double> interpValAtExtendedTau(const std::size_t spin, const std::size_t x1, const std::size_t x2, double tau) const;
    
    void interpValAtExtendedTau(const std::size_t spin, double tau, Eigen::Ref<Eigen::MatrixXcd> result) const;
    
    virtual void setParams(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau);
};







class BareGreenFunction : public GenericGreenFunction {
protected:
    Eigen::ArrayXXcd m_eiwt;
    
    void constructExpiwt();
    
public:
    BareGreenFunction() {}
    BareGreenFunction(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const std::size_t nt4eiwt, const MPI_Comm& comm = MPI_COMM_SELF);
    BareGreenFunction(const BareGreenFunction&) = default;
    BareGreenFunction(BareGreenFunction&&) = default;
    BareGreenFunction& operator=(const BareGreenFunction&) = default;
    BareGreenFunction& operator=(BareGreenFunction&&) = default;
    ~BareGreenFunction() = default;
    
    std::size_t tauGridSizeOfExpiwt() const {return m_eiwt.rows();}
    
    std::complex<double> expiwt(const std::size_t t, const std::size_t o) const {return m_eiwt(t, o);}
    
    void computeHighFreqExpan(const BareHamiltonian& H0);
    
    void invFourierTrans();
    
    void setParams(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const std::size_t nt4eiwt);
};






class GreenFunction : public GenericGreenFunction {
protected:
    // SqMatArray2XXd Gwvar;
    SqMatArray2XXcd m_S;
    Eigen::MatrixX2d m_dens;  // Accurate measure of spin- and site-resolved electron densities; dens = diag(G(beta-))
    Eigen::MatrixX2d m_densvar;  // Variance of measured densities
    
public:
    GreenFunction() {}
    GreenFunction(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const std::size_t nbins4S, const MPI_Comm& comm = MPI_COMM_SELF);
    GreenFunction(const GreenFunction&) = default;
    GreenFunction(GreenFunction&&) = default;
    GreenFunction& operator=(const GreenFunction&) = default;
    GreenFunction& operator=(GreenFunction&&) = default;
    ~GreenFunction() = default;
    
    // const SqMatArray2XXd& fCoeffsVar() const {return Gwvar;}
    // SqMatArray2XXd& fCoeffsVar() {return Gwvar;}
    
    const SqMatArray2XXcd& selfEnGF() const {return m_S;}
    SqMatArray2XXcd& selfEnGF() {return m_S;}
    
    std::size_t nTauBins4selfEgf() const {return m_S.dim1();}
    
    const Eigen::MatrixX2d& elecDensities() const {return m_dens;}
    Eigen::MatrixX2d& elecDensities() {return m_dens;}
    
    const Eigen::MatrixX2d& elecDensVars() const {return m_densvar;}
    Eigen::MatrixX2d& elecDensVars() {return m_densvar;}
    
    void computeHighFreqExpan(const BareHamiltonian& H0, const double U);
    
    double evalFromSelfEgf(const BareGreenFunction& G0);
    
    void setParams(const double beta, const std::size_t nc, const std::size_t nfcut, const std::size_t ntau, const std::size_t nbins4S);
};




// A simple approximation (with a finite smearing) to Dirc delta function
inline double delta(const double smear, const double x) {
    return (smear / 2) / (M_PI * (x * x + smear * smear / 4));
}


#endif /* green_function_hpp */
