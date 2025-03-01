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
    // Moments of high-frequency expansion of Green's function:
    // G(iw) = G0 / iw + G1 / (iw)^2 + G2 / (iw)^3 + ...
    // G0 = -[G(0+) - G(0-)] = -[G(0+) + G(beta-)] = -I, G1 = G'(0+) - G'(0-) = G'(0+) + G'(beta-), G2 = -[G"(0+) - G"(0-)] = -[G"(0+) + G"(beta-)].
    // Remember the nonstandard definition of Green's function here (negative of the standard definition, so G0 = -I instead of I here).
    SqMatArray23Xcd m_moms;
    CubicSplineMat2XXcd m_Gspl;  // Cubic spline for G(tau)
    
public:
    GenericGreenFunction() : m_beta(0.0) {}
    GenericGreenFunction(const double beta, const Eigen::Index nc, const Eigen::Index nfcut, const Eigen::Index ntau, const MPI_Comm& comm = MPI_COMM_SELF);
    GenericGreenFunction(const GenericGreenFunction&) = default;
    GenericGreenFunction(GenericGreenFunction&&) = default;
    GenericGreenFunction& operator=(const GenericGreenFunction&) = default;
    GenericGreenFunction& operator=(GenericGreenFunction&&) = default;
    
    virtual ~GenericGreenFunction() {}
    
    double inverseTemperature() const {return m_beta;}
    
    Eigen::Index tauGridSize() const {return m_Gt.dim1();}
    
    bool isDiscretized() const {return tauGridSize() > 1;}
    
    Eigen::Index freqCutoff() const {return m_Gw.dim1() - 1;}
    
    Eigen::Index nSites() const {return m_Gw.dimm();}
    
    const Eigen::ArrayXd& matsubFreqs() const {return m_matsfs;}
    const Eigen::ArrayXd& imagTimes() const {return m_imagts;}
    
    const SqMatArray2XXcd &fourierCoeffs() const {return m_Gw;}
    SqMatArray2XXcd &fourierCoeffs() {return m_Gw;}
    
    const SqMatArray2XXcd &valsOnTauGrid() const {return m_Gt;}
    SqMatArray2XXcd &valsOnTauGrid() {return m_Gt;}
    
    const SqMatArray23Xcd& moments() const {return m_moms;}
    // Return a const Eigen cwise-operation expression, whose type is too complicated to write
    const auto fCoeffHighFreq(const Eigen::Index spin, const std::complex<double> z) const {
        // We can safely return this Eigen expression because the moments are members of this class, not temporary objects within this function.
        const std::complex<double> zsq = z * z;
        return m_moms(spin, 0) / z + m_moms(spin, 1) / zsq + m_moms(spin, 2) / (zsq * z);
    }
    // Return a const Eigen cwise-operation expression, whose type is too complicated to write
    const auto valAtTauHighFreq(const Eigen::Index spin, const double tau) const {
        assert(tau >=0 && tau <= m_beta);
        // We can safely return this Eigen expression because G1 and G2 are members of this class, not temporary objects within this function
        return -0.5 * m_moms(spin, 0) + ((2.0 * tau - m_beta) / 4.0) * m_moms(spin, 1) + ((m_beta - tau) * tau / 4.0) * m_moms(spin, 2);
    }
    
    void fourierTransform();
    virtual void invFourierTrans();
    
    std::complex<double> valAtExtendedTau(const Eigen::Index spin, const Eigen::Index x1, const Eigen::Index x2, Eigen::Index ext_tau_ind,
                                          const LimitDirection approach) const;
    
    // Return a const Eigen cwise-operation expression, whose type is too complicated to write
    const auto valAtExtendedTau(const Eigen::Index spin, Eigen::Index ext_tau_ind, const LimitDirection approach) const;
    
    std::complex<double> interpValAtExtendedTau(const Eigen::Index spin, const Eigen::Index x1, const Eigen::Index x2, double tau) const;
    
    void interpValAtExtendedTau(const Eigen::Index spin, double tau, Eigen::Ref<Eigen::MatrixXcd> result) const;
    
    virtual void symmetrizeSpins(const bool gatherdatafirst = false);
    
    virtual void setParams(const double beta, const Eigen::Index nc, const Eigen::Index nfcut, const Eigen::Index ntau);
};







class BareGreenFunction : public GenericGreenFunction {
protected:
    Eigen::ArrayXXcd m_eiwt;
    
    void constructExpiwt();
    
public:
    BareGreenFunction() {}
    BareGreenFunction(const double beta, const Eigen::Index nc, const Eigen::Index nfcut, const Eigen::Index ntau, const Eigen::Index nt4eiwt, const MPI_Comm& comm = MPI_COMM_SELF);
    BareGreenFunction(const BareGreenFunction&) = default;
    BareGreenFunction(BareGreenFunction&&) = default;
    BareGreenFunction& operator=(const BareGreenFunction&) = default;
    BareGreenFunction& operator=(BareGreenFunction&&) = default;
    ~BareGreenFunction() = default;
    
    Eigen::Index tauGridSizeOfExpiwt() const {return m_eiwt.rows();}
    
    std::complex<double> expiwt(const Eigen::Index t, const Eigen::Index o) const {return m_eiwt(t, o);}
    
    void computeMoments(const BareHamiltonian& H0);
    
    void invFourierTrans();
    
    void setParams(const double beta, const Eigen::Index nc, const Eigen::Index nfcut, const Eigen::Index ntau, const Eigen::Index nt4eiwt);
};






class GreenFunction : public GenericGreenFunction {
protected:
    SqMatArray2XXd m_Gwvar;
    SqMatArray2XXcd m_S;
    Eigen::MatrixX2d m_dens;  // Accurate measure of spin- and site-resolved electron densities; dens = diag(G(beta-))
    Eigen::MatrixX2d m_densstddev;  // Variance of measured densities
    
public:
    double spinCorrelation;  // Intersite spin-spin correlation <Sz1 * Sz2>, a real number
    
    GreenFunction() {}
    GreenFunction(const double beta, const Eigen::Index nc, const Eigen::Index nfcut, const Eigen::Index ntau, const Eigen::Index nbins4S, const MPI_Comm& comm = MPI_COMM_SELF);
    GreenFunction(const GreenFunction&) = default;
    GreenFunction(GreenFunction&&) = default;
    GreenFunction& operator=(const GreenFunction&) = default;
    GreenFunction& operator=(GreenFunction&&) = default;
    ~GreenFunction() = default;
    
    const SqMatArray2XXd& fCoeffsVar() const {return m_Gwvar;}
    SqMatArray2XXd& fCoeffsVar() {return m_Gwvar;}
    
    const SqMatArray2XXcd& selfEnGF() const {return m_S;}
    SqMatArray2XXcd& selfEnGF() {return m_S;}
    
    Eigen::Index nTauBins4selfEnGF() const {return m_S.dim1();}
    
    const Eigen::MatrixX2d& densities() const {return m_dens;}
    Eigen::MatrixX2d& densities() {return m_dens;}
    
    const Eigen::MatrixX2d& densStdDev() const {return m_densstddev;}
    Eigen::MatrixX2d& densStdDev() {return m_densstddev;}
    
    void computeMoments(const BareHamiltonian& H0, const double U);
    
    double evalFromSelfEnGF(const BareGreenFunction& G0);
    
    void symmetrizeSpins(const bool gatherdatafirst = false);
    
    void setParams(const double beta, const Eigen::Index nc, const Eigen::Index nfcut, const Eigen::Index ntau, const Eigen::Index nbins4S);
};




// A simple approximation (with a finite smearing) to Dirc delta function
inline double delta(const double smear, const double x) {
    return (smear / 2) / (M_PI * (x * x + smear * smear / 4));
}


#endif /* green_function_hpp */
