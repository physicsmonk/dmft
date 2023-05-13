//
//  anal_continuation.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef pade_hpp
#define pade_hpp

#include <cmath>
#include <Eigen/QR>
#include <Eigen/LU>
#include "mpreal.h"
#include <unsupported/Eigen/MPRealSupport>
#include "bare_hamiltonian.hpp"
#include "gf_data_wrapper.hpp"

using namespace std::complex_literals;

// enum LSsolver : int {BDCSVD = 0, CompleteOrthogonalDecomposition = 1, ColPivHouseholderQR = 2};

template <typename _InnerRealType, int _n0, int _n1, int _nm>
class PadeApproximant {
public:
    // Assumes all the std::vector are sorted ascendingly. Only use n1 = 0 slice of selfen_static, which will be made a copy of
    PadeApproximant& build(const SqMatArray<std::complex<double>, _n0, _n1, _nm>& selfen_dyn_matsub, const double beta, const Eigen::ArrayXi& datalens,
                           const Eigen::ArrayXi& startfreqs, const Eigen::ArrayXi& coefflens, const MPI_Comm& comm = MPI_COMM_SELF);
    
    PadeApproximant() = default;
    PadeApproximant(const SqMatArray<std::complex<double>, _n0, _n1, _nm>& selfen_dyn_matsub, const double beta, const Eigen::ArrayXi& datalens,
                    const Eigen::ArrayXi& startfreqs, const Eigen::ArrayXi& coefflens, const MPI_Comm& comm = MPI_COMM_SELF) {
        build(selfen_dyn_matsub, beta, datalens, startfreqs, coefflens, comm);
    }
    PadeApproximant(const PadeApproximant&) = default;
    PadeApproximant(PadeApproximant&&) = default;
    PadeApproximant& operator=(const PadeApproximant&) = default;
    PadeApproximant& operator=(PadeApproximant&&) = default;
    ~PadeApproximant() = default;
    
    void computeSpectra(const SqMatArray<std::complex<double>, _n0, 1, _nm>& selfen_static, const BareHamiltonian& H0, const Eigen::Index np, const double low,
                        const double high, const double del, const bool physonly = true);
    
    const Eigen::Array<int, _n0, 1>& nPhysSpectra() const {return m_nphys;}
    const SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm>& spectraMatrix() const {return m_spectramat;}
    const SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm>& retardedSelfEn() const {return m_selfenR;}
    
private:
    //MPI_Comm m_comm;
    //int m_prank, m_psize;
    //Eigen::Index m_napproxs, m_dim0, m_dimm;  // Store the relevant dimensions
    // Stores Pade coefficients. Must use a std::vector because the coefficients of different Pade approximants have different lengths.
    // Each Eigen::Vector contains the coefficients of a Pade approximant. The index of the std::vector is site, then spin, and then
    // approximant, major.
    std::vector<Eigen::Vector<std::complex<_InnerRealType>, Eigen::Dynamic> > m_coeffs;
    Eigen::Array<int, _n0, 1> m_nphys;  // Stores the number of physical approximants for each index of _dim0
    // Stores the analytically continued (with positive delta) self-energy and the spectra matrix = (G(w + i * del) - G(w - i * del)) / (2 * pi * i)
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_selfenR, m_spectramat;
};

template <typename _InnerRealType, int _n0, int _n1, int _nm>
PadeApproximant<_InnerRealType, _n0, _n1, _nm>& PadeApproximant<_InnerRealType, _n0, _n1, _nm>::build(const SqMatArray<std::complex<double>, _n0, _n1, _nm>& selfen_dyn_matsub, const double beta, const Eigen::ArrayXi& datalens, const Eigen::ArrayXi& startfreqs, const Eigen::ArrayXi& coefflens, const MPI_Comm& comm) {
    const Eigen::Index n0 = selfen_dyn_matsub.dim0();
    const Eigen::Index nm = selfen_dyn_matsub.dimm();
    if (startfreqs.maxCoeff() + datalens.maxCoeff() > selfen_dyn_matsub.dim1() || -startfreqs.minCoeff() - 1 >= selfen_dyn_matsub.dim1())
        throw std::range_error("Required data length exceeds the used data length for building Pade approximant!");
    
    Eigen::ArrayXi::const_iterator itM, itn0, itN;
    int iz, n;
    Eigen::Index r, s, x0, x1, r0, ir;
    Eigen::Matrix<std::complex<_InnerRealType>, Eigen::Dynamic, Eigen::Dynamic> A, F(0, n0 * nm * nm);
    Eigen::Vector<std::complex<_InnerRealType>, Eigen::Dynamic> zs, b;
    
    int prank, psize;
    MPI_Comm_rank(comm, &prank);
    MPI_Comm_size(comm, &psize);
    
    // Most evenly distribute tasks to each process, treating the three data dimensions on equal footing,
    // because the first data dimension could be small (around 10) so it is not so efficient when using
    // moderately many processes if just dividing the first data dimension
    //const Eigen::Index totalsize = datalens.size() * startfreqs.size() * coefflens.size();
    const Eigen::Index n0Nsize = startfreqs.size() * coefflens.size();
    //const Eigen::Index bbsize = totalsize / m_psize;
    //Eigen::Index localstart = m_prank * bbsize;
    //Eigen::Index localfinal = (m_prank < m_psize - 1) ? localstart + bbsize : totalsize;
    Eigen::Index localstart, localfinal;
    mostEvenPart(datalens.size() * startfreqs.size() * coefflens.size(), psize, prank, localfinal, localstart);  // localfinal now is the local size
    localfinal += localstart;  // Now obtain localfinal
    const Eigen::Index localMstart = localstart / n0Nsize;
    const Eigen::Index localMfinal = localfinal / n0Nsize + (localfinal % n0Nsize > 0);
    localstart %= n0Nsize;
    localfinal %= n0Nsize;
    const Eigen::Index localn0start = localstart / coefflens.size();
    const Eigen::Index localn0final = localfinal == 0 ? startfreqs.size() : localfinal / coefflens.size() + (localfinal % coefflens.size() > 0);
    localstart %= coefflens.size();
    localfinal %= coefflens.size();
    const Eigen::Index localNstart = localstart;
    const Eigen::Index localNfinal = localfinal == 0 ? coefflens.size() : localfinal;
    
    const Eigen::ArrayXi::const_iterator localMbegin = datalens.cbegin() + localMstart;
    const Eigen::ArrayXi::const_iterator localMend = datalens.cbegin() + localMfinal;
    const Eigen::ArrayXi::const_iterator localn0begin = startfreqs.cbegin() + localn0start;
    const Eigen::ArrayXi::const_iterator localn0end = startfreqs.cbegin() + localn0final;
    const Eigen::ArrayXi::const_iterator localNbegin = coefflens.cbegin() + localNstart;
    const Eigen::ArrayXi::const_iterator localNend = coefflens.cbegin() + localNfinal;
    Eigen::ArrayXi::const_iterator n0end, Nend;
    
    // Test
    // std::cout << "Rank " << _prank << ": localMstart = " << localMstart << ", localMfinal = " << localMfinal << "; localn0start = " << localn0start
    // << ", localn0final = " << localn0final << "; localNstart = " << localNstart << ", localNfinal = " << localNfinal << std::endl;
    
    // Empty solvers
#define PADE_LS_SOLVER 0
    //Eigen::JacobiSVD<Eigen::Matrix<std::complex<_InnerFloatType>, Eigen::Dynamic, Eigen::Dynamic> > jacobisvd;  // Optimally accurate but very slow
#if PADE_LS_SOLVER == 1
    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<std::complex<_InnerRealType>, Eigen::Dynamic, Eigen::Dynamic> > lssolver;  // Faster but less accurate than BDCSVD
#elif PADE_LS_SOLVER == 2
    Eigen::ColPivHouseholderQR<Eigen::Matrix<std::complex<_InnerRealType>, Eigen::Dynamic, Eigen::Dynamic> > lsssolver;  // Faster but less accurate than BDCSVD
#else
    Eigen::BDCSVD<Eigen::Matrix<std::complex<_InnerRealType>, Eigen::Dynamic, Eigen::Dynamic> > lssolver;  // Most accurate but a little bit slow
#endif
    
    m_coeffs.clear();
    for (itM = localMbegin; itM < localMend; ++itM) {
        // Test
        //std::cout << "Rank " << prank << ": M = " << *itM << std::endl;
        zs.resize(*itM);
        F.resize(*itM, Eigen::NoChange);
        n0end = itM == localMend - 1 ? localn0end : startfreqs.cend();
        for (itn0 = itM == localMbegin ? localn0begin : startfreqs.cbegin(); itn0 < n0end; ++itn0) {
            // Test
            // std::cout << "Rank " << _prank << ": n0 = " << *itn0 << std::endl;
            for (iz = 0; iz < *itM; ++iz) zs(iz) = std::complex<_InnerRealType>(0.0, (2 * (*itn0 + iz) + 1) * M_PI / beta);  // Assemble the z array
            // Assemble the f(z) array
            for (s = 0; s < n0; ++s) {
                for (x1 = 0; x1 < nm; ++x1) {
                    for (x0 = 0; x0 < nm; ++x0) {
                        for (iz = 0; iz < *itM; ++iz) {
                            n = *itn0 + iz;
                            // Sequentially (column-majorly) access the matrix element
                            if (n < 0) F(iz, (s * nm + x1) * nm + x0) = std::conj(selfen_dyn_matsub(s, -n - 1, x1, x0));
                            else F(iz, (s * nm + x1) * nm + x0) = selfen_dyn_matsub(s, n, x0, x1);
                        }
                    }
                }
            }
            A = Eigen::Matrix<std::complex<_InnerRealType>, Eigen::Dynamic, Eigen::Dynamic>::Ones(*itM, 2);
            Nend = itM == localMend - 1 && itn0 == localn0end - 1 ? localNend : coefflens.cend();
            for (itN = itM == localMbegin && itn0 == localn0begin ? localNbegin : coefflens.cbegin();
                 itN < Nend && *itN <= *itM;
                 ++itN) {
                // Test
                //std::cout << "Rank " << prank << ": N = " << *itN << std::endl;
                if (*itN % 2 != 0) throw std::invalid_argument("The number of Pade coefficients must be even!");
                r0 = A.cols() / 2;
                r = *itN / 2;
                A.conservativeResize(Eigen::NoChange, *itN);
                // Assemble the left-half columns of coefficient matrix of the linear least square system
                for (ir = r0; ir < r; ++ir) A.col(ir) = A.col(ir - 1).cwiseProduct(zs);
                for (s = 0; s < n0; ++s) {
                    for (x1 = 0; x1 < nm; ++x1) {
                        for (x0 = 0; x0 < nm; ++x0) {
                            A.rightCols(r).noalias() = F.col((s * nm + x1) * nm + x0).asDiagonal() * (-A.leftCols(r));
                            // Assemble the rhs vector of the linear least square system
                            b = -A.col(*itN - 1).cwiseProduct(zs);
                            // Solve for and store the unfiltered Pade coefficients
                            // BDCSVD should be compiled without unsafe math optimizations, e.g., for Intel's compiler, compile with -fp-model precise option
#if PADE_LS_SOLVER == 1 || PADE_LS_SOLVER == 2
                            m_coeffs.push_back(lssolver.compute(A).solve(b));
#else
                            m_coeffs.push_back(lssolver.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b));
#endif
                            
                            // m_coeffs.push_back((A.adjoint() * A).ldlt().solve(A.adjoint() * b));
                            
                            // Test
                            // std::cout << "Rank " << _prank << ": Check point 3: " << *itN << ", " << s << ", " << x << ", " << *itM << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    // set MPI communicator for data members
    m_selfenR.mpiCommunicator(comm);   // For future mpi sum
    m_spectramat.mpiCommunicator(comm);
    
    // Test
    // std::cout << "Rank " << _prank << ": Check point 4: " << _napproxs << std::endl;
    return *this;
}

template <typename _InnerRealType, int _n0, int _n1, int _nm>
void PadeApproximant<_InnerRealType, _n0, _n1, _nm>::computeSpectra(const SqMatArray<std::complex<double>, _n0, 1, _nm>& selfen_static, const BareHamiltonian& H0,
                                                                    const Eigen::Index np, const double low, const double high, const double del, const bool physonly) {
    assert(np > 1);
    
    const Eigen::Index n0 = selfen_static.dim0();
    const Eigen::Index nm = selfen_static.dimm();
    const Eigen::Index nmatelems = n0 * nm * nm;
    const Eigen::Index napproxs = m_coeffs.size() / nmatelems;
    Eigen::Index i, s, x0, x1, k, o, r0, r, ir;
    std::complex<_InnerRealType> SigmaR;
    Eigen::Vector<std::complex<_InnerRealType>, Eigen::Dynamic> zs = Eigen::Vector<_InnerRealType, Eigen::Dynamic>::LinSpaced(np, low, high)
    + std::complex<_InnerRealType>(0.0, std::fabs(del)) * Eigen::Vector<_InnerRealType, Eigen::Dynamic>::Ones(np);
    Eigen::Matrix<std::complex<_InnerRealType>, Eigen::Dynamic, Eigen::Dynamic> zpol = Eigen::Matrix<_InnerRealType, Eigen::Dynamic, Eigen::Dynamic>::Ones(np, 1);
    // Purely local; will be used to temporarily store the static part of the self-energy
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, Eigen::Dynamic> selfenR(1, np, nm);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(nm);
    bool is_physical;
    
    m_nphys = Eigen::ArrayXi::Zero(n0);
    m_selfenR.resize(n0, np, nm);
    m_selfenR().setZero();
    
    for (i = 0; i < napproxs; ++i) {
        r0 = zpol.cols();
        r = m_coeffs[i * nmatelems].size() / 2;   // Next nmatelems coefficient vector have the same size
        if (r > r0) {
            zpol.conservativeResize(Eigen::NoChange, r);
            for (ir = r0; ir < r; ++ir) zpol.col(ir) = zpol.col(ir - 1).cwiseProduct(zs);
        }
        for (s = 0; s < n0; ++s) {
            is_physical = true;
            for (o = 0; o < np; ++o) {
                for (x1 = 0; x1 < nm; ++x1) {
                    for (x0 = 0; x0 < nm; ++x0) {
                        k = i * nmatelems + (s * nm + x1) * nm + x0;
                        // zpol.row(o) could be longer than r
                        SigmaR = (zpol.row(o).head(r) * m_coeffs[k].head(r))(0) / ((zpol.row(o).head(r) * m_coeffs[k].tail(r))(0) + zpol(o, r - 1) * zs(o));
                        if constexpr (std::is_same<_InnerRealType, mpfr::mpreal>::value) {
                            // Can only assign to double complex like this because no direct conversion to double complex
                            selfenR(0, o, x0, x1).real(SigmaR.real().toDouble());
                            selfenR(0, o, x0, x1).imag(SigmaR.imag().toDouble());
                        }
                        else selfenR(0, o, x0, x1) = static_cast<std::complex<double> >(SigmaR);
                    }
                }
                if (selfenR[o].array().isNaN().any()) {
                    is_physical = false;
                    break;
                }
                else if (physonly) {
                    // Checks causality by checking if the Hermitian matrix (selfen - selfen^H) / 2i is negative semi-definite.
                    // If not, mark the corresponding approximant as unphysical.
                    es.compute((selfenR[o] - selfenR[o].adjoint()) / 2i, Eigen::EigenvaluesOnly);
                    //if ( (((selfenR[o] - selfenR[o].adjoint()) / 2i).selfadjointView<Eigen::Lower>().eigenvalues().array() > 0.0).any() ) {
                    if ((es.eigenvalues().array() > 0.0).any()) {
                        is_physical = false;
                        break;
                    }
                }
            }
            if (is_physical) {
                m_nphys(s) += 1;
                for (o = 0; o < np; ++o) m_selfenR(s, o) += selfenR[o];
            }
        }
    }
    
    MPI_Allreduce(MPI_IN_PLACE, m_nphys.data(), m_nphys.size(), MPI_INT, MPI_SUM, m_selfenR.mpiCommunicator());
//    _selfen.sumLocals2mastPart();
//    std::array<Eigen::Index, 2> so;
//    for (i = 0; i < _selfen.mastPartSize(); ++i) {
//        so = _selfen.index2DinPart(i);
//        if (_nphys(so[0]) > 0) _selfen.masteredPart(i) /= _nphys(so[0]);
//    }
    m_selfenR.allSum();   // Sum results from all Pade approximants on all processes; full-size m_selfenR will be used by all processes
    for (s = 0; s < n0; ++s) m_selfenR.atDim0(s) /= m_nphys(s);
    
    m_spectramat.resize(n0, np, nm);
    computeLattGFfCoeffs(H0, m_selfenR, selfen_static, zs, m_spectramat);  // Only compute mastered partition of m_spectramat
    auto specmatmastpart = m_spectramat.mastFlatPart();
    // Call eval() to evaluate to a temporary to resolve the aliasing issue
    for (i = 0; i < specmatmastpart.size(); ++i) specmatmastpart[i] = (specmatmastpart[i] - specmatmastpart[i].adjoint().eval()) / (2i * M_PI);
    specmatmastpart.allGather();   // Make the full data available to all processes, although not necessary
    
    // Add static part back to the analytically-continued self-energy
    for (s = 0; s < n0; ++s) {
        for (o = 0; o < np; ++o) {
            m_selfenR(s, o) += selfen_static[s];
        }
    }
}



typedef PadeApproximant<long double, 2, Eigen::Dynamic, Eigen::Dynamic> PadeApproximant2XXld;
typedef PadeApproximant<mpfr::mpreal, 2, Eigen::Dynamic, Eigen::Dynamic> PadeApproximant2XXmpreal;


#endif /* pade_hpp */
