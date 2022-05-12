//
//  anal_continuation.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef anal_continuation_hpp
#define anal_continuation_hpp

#include <cmath>
#include <Eigen/QR>
#include <Eigen/LU>
#include "bare_hamiltonian.hpp"
#include "gf_data_wrapper.hpp"

using namespace std::complex_literals;


template <int n0, int n1, int nm>
class PadeApproximant {
public:
    // Assumes all the std::vector are sorted ascendingly. Pass pointer to selfen_static not const reference because
    // selfen_static will be accessed later on so we want passing const temporaries trigger compliation error.
    void build(const SqMatArray<std::complex<double>, n0, n1, nm>& selfen_matsub, const SqMatArray<std::complex<double>, n0, 1, nm> *selfen_static, const double beta, const Eigen::ArrayXi& datalens, const Eigen::ArrayXi& startfreqs, const Eigen::ArrayXi& coefflens, const MPI_Comm& comm = MPI_COMM_SELF);
    
    PadeApproximant() : m_ptr2selfenstatic(nullptr) {}   // Inline default constructor
    PadeApproximant(const SqMatArray<std::complex<double>, n0, n1, nm>& selfen_matsub, const SqMatArray<std::complex<double>, n0, 1, nm> *selfen_static, const double beta, const Eigen::ArrayXi& datalens, const Eigen::ArrayXi& startfreqs, const Eigen::ArrayXi& coefflens, const MPI_Comm& comm = MPI_COMM_SELF) {build(selfen_matsub, selfen_static, beta, datalens, startfreqs, coefflens, comm);}
    PadeApproximant(const PadeApproximant&) = default;
    PadeApproximant(PadeApproximant&&) = default;
    PadeApproximant& operator=(const PadeApproximant&) = default;
    PadeApproximant& operator=(PadeApproximant&&) = default;
    ~PadeApproximant() = default;
    
    void computeSpectra(const BareHamiltonian& H0, const std::size_t np, const double low, const double high, const double del, const bool physonly = true);
    
    const Eigen::Array<int, n0, 1>& nPhysSpectra() const;
    
    const SqMatArray<std::complex<double>, n0, Eigen::Dynamic, nm>& spectraMatrix() const;
    
    const SqMatArray<std::complex<double>, n0, Eigen::Dynamic, nm>& retardedSelfEn() const;
    
private:
    typedef long double m_Hpfloat;   // Could modify this to implement even higher precision floating-point type for solving the linear least-square systems
    const SqMatArray<std::complex<double>, n0, 1, nm> *m_ptr2selfenstatic;  // Just a link to const external static part of self-energy
    //MPI_Comm m_comm;
    //int m_prank, m_psize;
    //std::size_t m_napproxs, m_dim0, m_dimm;  // Store the relevant dimensions
    // Stores Pade coefficients. Must use a std::vector because the coefficients of different Pade approximants have different lengths.
    // Each Eigen::Vector contains the coefficients of a Pade approximant. The index of the std::vector is site, then spin, and then
    // approximant, major.
    std::vector<Eigen::Vector<std::complex<m_Hpfloat>, Eigen::Dynamic> > m_coeffs;
    Eigen::Array<int, n0, 1> m_nphys;  // Stores the number of physical approximants for each index of _dim0
    // Stores the analytically continued (with positive delta) self-energy and the spectra matrix = (G(w + i * del) - G(w - i * del)) / (2 * pi * i)
    SqMatArray<std::complex<double>, n0, Eigen::Dynamic, nm> m_selfenR, m_spectramat;
};

template <int n0, int n1, int nm>
void PadeApproximant<n0, n1, nm>::build(const SqMatArray<std::complex<double>, n0, n1, nm>& selfen_matsub, const SqMatArray<std::complex<double>, n0, 1, nm> *selfen_static, const double beta, const Eigen::ArrayXi& datalens, const Eigen::ArrayXi& startfreqs, const Eigen::ArrayXi& coefflens, const MPI_Comm& comm) {
    if (startfreqs.maxCoeff() + datalens.maxCoeff() > static_cast<int>(selfen_matsub.dim1()) || -startfreqs.minCoeff() - 1 >= static_cast<int>(selfen_matsub.dim1()))
        throw std::range_error("Required data length exceeds the used data length for building Pade approximant!");
    
    m_ptr2selfenstatic = selfen_static;  // Link to external static part of self-energy
    
    Eigen::ArrayXi::const_iterator itM, itn0, itN;
    int iz, n;
    std::size_t r, s, x0, x1, r0, ir;
    Eigen::Matrix<std::complex<m_Hpfloat>, Eigen::Dynamic, Eigen::Dynamic> A, F(0, selfen_matsub.dim0() * selfen_matsub.dimm() * selfen_matsub.dimm());
    Eigen::Vector<std::complex<m_Hpfloat>, Eigen::Dynamic> zs, b;
    
    int prank, psize;
    MPI_Comm_rank(comm, &prank);
    MPI_Comm_size(comm, &psize);
    
    // Most evenly distribute tasks to each process, treating the three data dimensions on equal footing,
    // because the first data dimension could be small (around 10) so it is not so efficient when using
    // moderately many processes if just dividing the first data dimension
    //const std::size_t totalsize = datalens.size() * startfreqs.size() * coefflens.size();
    const std::size_t n0Nsize = startfreqs.size() * coefflens.size();
    //const std::size_t bbsize = totalsize / m_psize;
    //std::size_t localstart = m_prank * bbsize;
    //std::size_t localfinal = (m_prank < m_psize - 1) ? localstart + bbsize : totalsize;
    std::size_t localstart, localfinal;
    mostEvenPart(datalens.size() * startfreqs.size() * coefflens.size(), psize, prank, localfinal, localstart);  // localfinal now is the local size
    localfinal += localstart;  // Now obtain localfinal
    const std::size_t localMstart = localstart / n0Nsize;
    const std::size_t localMfinal = localfinal / n0Nsize + (localfinal % n0Nsize > 0);
    localstart %= n0Nsize;
    localfinal %= n0Nsize;
    const std::size_t localn0start = localstart / coefflens.size();
    const std::size_t localn0final = localfinal == 0 ? startfreqs.size() : localfinal / coefflens.size() + (localfinal % coefflens.size() > 0);
    localstart %= coefflens.size();
    localfinal %= coefflens.size();
    const std::size_t localNstart = localstart;
    const std::size_t localNfinal = localfinal == 0 ? coefflens.size() : localfinal;
    
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
    
    Eigen::BDCSVD<Eigen::Matrix<std::complex<m_Hpfloat>, Eigen::Dynamic, Eigen::Dynamic> > bdcsvd;
    //Eigen::JacobiSVD<Eigen::Matrix<std::complex<m_Hpfloat>, Eigen::Dynamic, Eigen::Dynamic> > jacobisvd;
    
    m_coeffs.clear();
    for (itM = localMbegin; itM < localMend; ++itM) {
        // Test
        // std::cout << "Rank " << _prank << ": M = " << *itM << std::endl;
        zs.resize(*itM);
        F.resize(*itM, Eigen::NoChange);
        n0end = itM == localMend - 1 ? localn0end : startfreqs.cend();
        for (itn0 = itM == localMbegin ? localn0begin : startfreqs.cbegin(); itn0 < n0end; ++itn0) {
            // Test
            // std::cout << "Rank " << _prank << ": n0 = " << *itn0 << std::endl;
            for (iz = 0; iz < *itM; ++iz) zs(iz) = 1il * ((2.0l * (*itn0 + iz) + 1.0l) * M_PI / beta);  // Assemble the z array
            // Assemble the f(z) array
            for (s = 0; s < selfen_matsub.dim0(); ++s) {
                for (x1 = 0; x1 < selfen_matsub.dimm(); ++x1) {
                    for (x0 = 0; x0 < selfen_matsub.dimm(); ++x0) {
                        for (iz = 0; iz < *itM; ++iz) {
                            n = *itn0 + iz;
                            // Sequentially (column-majorly) access the matrix element
                            if (n < 0) F(iz, (s * selfen_matsub.dimm() + x1) * selfen_matsub.dimm() + x0) = std::conj(selfen_matsub(s, -n - 1, x1, x0) - (*selfen_static)(s, 0, x1, x0));
                            else F(iz, (s * selfen_matsub.dimm() + x1) * selfen_matsub.dimm() + x0) = selfen_matsub(s, n, x0, x1) - (*selfen_static)(s, 0, x0, x1);
                        }
                    }
                }
            }
            A = Eigen::Matrix<std::complex<m_Hpfloat>, Eigen::Dynamic, Eigen::Dynamic>::Ones(*itM, 2);
            Nend = itM == localMend - 1 && itn0 == localn0end - 1 ? localNend : coefflens.cend();
            for (itN = itM == localMbegin && itn0 == localn0begin ? localNbegin : coefflens.cbegin();
                 itN < Nend && *itN <= *itM;
                 ++itN) {
                // Test
                // std::cout << "Rank " << _prank << ": N = " << *itN << std::endl;
                if (*itN % 2 != 0) throw std::invalid_argument("The number of Pade coefficients must be even!");
                r0 = A.cols() / 2;
                r = *itN / 2;
                A.conservativeResize(Eigen::NoChange, *itN);
                // Assemble the left-half columns of coefficient matrix of the linear least square system
                for (ir = r0; ir < r; ++ir) A.col(ir) = A.col(ir - 1).cwiseProduct(zs);
                for (s = 0; s < selfen_matsub.dim0(); ++s) {
                    for (x1 = 0; x1 < selfen_matsub.dimm(); ++x1) {
                        for (x0 = 0; x0 < selfen_matsub.dimm(); ++x0) {
                            A.rightCols(r).noalias() = F.col((s * selfen_matsub.dimm() + x1) * selfen_matsub.dimm() + x0).asDiagonal() * (-A.leftCols(r));
                            // Assemble the rhs vector of the linear least square system
                            b = -A.col(*itN - 1).cwiseProduct(zs);
                            // Solve for and store the unfiltered Pade coefficients
                            // BDCSVD should be compiled without unsafe math optimizations, e.g., for Intel's compiler, compile with -fp-model precise option
                            bdcsvd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
                            m_coeffs.push_back(bdcsvd.solve(b));
                            // m_coeffs.push_back(A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b));  // Most accurate but a little bit slow
                            // m_coeffs.push_back(A.completeOrthogonalDecomposition().solve(b));  // Faster but less accurate than BDCSVD
                            // m_coeffs.push_back(A.colPivHouseholderQr().solve(b));  // Faster but less accurate than BDCSVD
                            // m_coeffs.push_back((A.adjoint() * A).ldlt().solve(A.adjoint() * b));
                            // _coeffs.push_back((A.adjoint() * A).partialPivLu().solve(A.adjoint() * b));
                            // _coeffs.push_back((A.adjoint() * A).fullPivLu().solve(A.adjoint() * b));
                            
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
}

template <int n0, int n1, int nm>
void PadeApproximant<n0, n1, nm>::computeSpectra(const BareHamiltonian& H0, const std::size_t np, const double low, const double high, const double del, const bool physonly) {
    assert(np > 1);
    
    const std::size_t nmatelems = m_ptr2selfenstatic->dim0() * m_ptr2selfenstatic->dimm() * m_ptr2selfenstatic->dimm();
    const std::size_t napproxs = m_coeffs.size() / nmatelems;
    std::size_t i, s, x0, x1, k, o, r0, r, ir;
    std::complex<m_Hpfloat> pol0, pol1;
    Eigen::Vector<std::complex<m_Hpfloat>, Eigen::Dynamic> zs(np);
    Eigen::Matrix<std::complex<m_Hpfloat>, Eigen::Dynamic, Eigen::Dynamic> zpol;
    // Purely local; will be used to temporarily store the static part of the self-energy
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, Eigen::Dynamic> selfenR(1, np, m_ptr2selfenstatic->dimm());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(m_ptr2selfenstatic->dimm());
    bool is_physical;
    
    m_nphys = Eigen::ArrayXi::Zero(m_ptr2selfenstatic->dim0());
    m_selfenR.resize(m_ptr2selfenstatic->dim0(), np, m_ptr2selfenstatic->dimm());
    m_selfenR().setZero();
    
    // zs has positive delta no matter the sign of the input del
    zs = Eigen::Vector<std::complex<m_Hpfloat>, Eigen::Dynamic>::LinSpaced(np, low, high) + 1i * std::fabs(del) * Eigen::Vector<std::complex<m_Hpfloat>, Eigen::Dynamic>::Ones(np);
    zpol = Eigen::Matrix<std::complex<m_Hpfloat>, Eigen::Dynamic, Eigen::Dynamic>::Ones(np, 1);
    
    for (i = 0; i < napproxs; ++i) {
        r0 = zpol.cols();
        r = m_coeffs[i * nmatelems].size() / 2;   // Next nmatelems coefficient vector have the same size
        if (r > r0) {
            zpol.conservativeResize(Eigen::NoChange, r);
            for (ir = r0; ir < r; ++ir) zpol.col(ir) = zpol.col(ir - 1).cwiseProduct(zs);
        }
        for (s = 0; s < m_selfenR.dim0(); ++s) {
            is_physical = true;
            for (o = 0; o < np; ++o) {
                for (x1 = 0; x1 < m_selfenR.dimm(); ++x1) {
                    for (x0 = 0; x0 < m_selfenR.dimm(); ++x0) {
                        k = i * nmatelems + (s * m_selfenR.dimm() + x1) * m_selfenR.dimm() + x0;
                        pol0 = (zpol.row(o).head(r) * m_coeffs[k].head(r))(0);  // zpol.row(o) could be longer than r
                        pol1 = (zpol.row(o).head(r) * m_coeffs[k].tail(r))(0);
                        selfenR(0, o, x0, x1) = static_cast<std::complex<double> >(pol0 / (pol1 + zpol(o, r - 1) * zs(o)));
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
    
    MPI_Allreduce(MPI_IN_PLACE, m_nphys.data(), static_cast<int>(m_nphys.size()), MPI_INT, MPI_SUM, m_selfenR.mpiCommunicator());
//    _selfen.sumLocals2mastPart();
//    std::array<std::size_t, 2> so;
//    for (i = 0; i < _selfen.mastPartSize(); ++i) {
//        so = _selfen.index2DinPart(i);
//        if (_nphys(so[0]) > 0) _selfen.masteredPart(i) /= _nphys(so[0]);
//    }
    m_selfenR.allSum();   // Sum results from all Pade approximants on all processes; full-size m_selfenR will be used by all processes
    for (s = 0; s < m_selfenR.dim0(); ++s) {
        for (o = 0; o < np; ++o) {
            if (m_nphys(s) > 0) m_selfenR(s, o) /= m_nphys(s);
            m_selfenR(s, o) += (*m_ptr2selfenstatic)[s];  // Add static part back to the analytically-continued self-energy
        }
    }
    
    m_spectramat.resize(m_selfenR.dim0(), np, m_selfenR.dimm());
    computeLattGFfCoeffs(H0, m_selfenR, zs, m_spectramat);  // Only compute mastered partition of m_spectramat
    auto specmatmastpart = m_spectramat.mastFlatPart();
    // Call eval() to evaluate to a temporary to resolve the aliasing issue
    for (i = 0; i < specmatmastpart.size(); ++i) specmatmastpart[i] = (specmatmastpart[i] - specmatmastpart[i].adjoint().eval()) / (2i * M_PI);
    specmatmastpart.allGather();   // Make the full data available to all processes, although not necessary
}

template <int n0, int n1, int nm>
inline const Eigen::Array<int, n0, 1>& PadeApproximant<n0, n1, nm>::nPhysSpectra() const {
    return m_nphys;
}

template <int n0, int n1, int nm>
inline const SqMatArray<std::complex<double>, n0, Eigen::Dynamic, nm>& PadeApproximant<n0, n1, nm>::spectraMatrix() const {
    return m_spectramat;
}

template <int n0, int n1, int nm>
inline const SqMatArray<std::complex<double>, n0, Eigen::Dynamic, nm>& PadeApproximant<n0, n1, nm>::retardedSelfEn() const {
    return m_selfenR;
}




typedef PadeApproximant<2, Eigen::Dynamic, Eigen::Dynamic> PadeApproximant2XX;


#endif /* anal_continuation_hpp */
