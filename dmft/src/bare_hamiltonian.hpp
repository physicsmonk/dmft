//
//  bare_hamiltonian.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef bare_hamiltonian_hpp
#define bare_hamiltonian_hpp

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
//#include "mpreal.h"
#include "gf_data_wrapper.hpp"

using namespace std::complex_literals;


// Base class for user-defined bare Hamiltonians. Users want to implement the virtual method
// constructHamiltonian in derived classes to construct any Hamiltonian matrix they want.
class BareHamiltonian {
private:
    MPI_Comm m_comm;
    int m_psize, m_prank;
    Eigen::Index m_klocalsize, m_klocalstart;
    double m_v0;   //  Unit cell volume/area/length
    Eigen::MatrixXd m_K;  // Stores reciprocal primative vectors in columns
    ArrayXindex m_nk;   // Numbers of k-points along each reciprocal primative vector
    std::array<double, 2> m_erange;   // Energy range of the band structure
    Eigen::ArrayXXd m_bands, m_bandpath;  // Stores energy bands; index is of (energy, (kx, ky, kz))
    // Block diagonalized Hamiltonian for the special case, 2D dimer Hubbard model in magnetic fields. First index runs over k-vectors (ky major)
    // and the second index runs over the eigenvalue space of the block diagonalization.
    SqMatArray<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, 2> m_HdimerMag2d;
    // Fermi velocity matrices in the space of _HdimerMag2d for the special case, 2D dimer Hubbard model in magnetic fields.
    // Local-size: first index is just x or y, second index runs over local k-vectors (ky major)
    SqMatArray2XXcd m_vdimerMag2d;
    std::string m_type;
    Eigen::ArrayX2d m_dos;   // Stores density of states
    double m_mu;  // Chemical potential. Note band structure and DOS are independent of it, but we set zero energy of bands along path to chemical potential
    SqMatArray22Xcd m_moments;  // First and second moments
    
protected:
    Eigen::MatrixXd m_a;  // Stores primative vectors in columns
    Eigen::ArrayXcd m_t;  // Stores hopping matrix elements
    
public:
    // No need for an explicit constructor, which is also user-friendly
    virtual ~BareHamiltonian() {}
    
    // Provide a method for index converting for k-space storage, otherwise we could make a data wrapper
    Eigen::Index flatIndex(const Eigen::Index ix, const Eigen::Index iy, const Eigen::Index iz) const {
        if (m_a.cols() != 3) throw std::bad_function_call();
        return (ix * m_nk(1) + iy) * m_nk(2) + iz;
    }
    Eigen::Index flatIndex(const Eigen::Index ix, const Eigen::Index iy) const {
        if (m_a.cols() != 2) throw std::bad_function_call();
        return ix * m_nk(1) + iy;
    }
    template <typename Derived>
    Eigen::Index flatIndex(const Eigen::DenseBase<Derived>& idvec) const {
        Eigen::Index id = idvec(0);
        for (Eigen::Index j = 1; j < idvec.size(); ++j) id = id * m_nk(j) + idvec(j);
        return id;
    }
    
    template <typename Derived, typename OtherDerived>
    void kVecAtIdVec(const Eigen::ArrayBase<Derived>& idvec, const Eigen::MatrixBase<OtherDerived>& k) const {
        Eigen::MatrixBase<OtherDerived>& k_ = const_cast<Eigen::MatrixBase<OtherDerived>&>(k);
        k_ = (m_K * (idvec.template cast<double>() / m_nk.cast<double>() - 0.5).matrix().asDiagonal()).rowwise().sum();
    }
    template <typename Derived>
    void kVecAtFlatId(Eigen::Index id, const Eigen::MatrixBase<Derived>& k) const {  // Calculate the ik-th k vector
        //    if (_K.rows() == 1) k = static_cast<double>(ik) / _nk(0) * _K.col(0);
        //    else if (_K.rows() == 2) {
        //        const Eigen::Index ix = ik / _nk(1);
        //        const Eigen::Index iy = ik % _nk(1);
        //        k = static_cast<double>(ix) / _nk(0) * _K.col(0) + static_cast<double>(iy) / _nk(1) * _K.col(1);
        //    }
        //    else if (_K.rows() == 3) {
        //        const Eigen::Index nk12 = _nk(1) * _nk(2);
        //        const Eigen::Index ix = ik / nk12;
        //        const Eigen::Index iy = (ik % nk12) / _nk(2);
        //        const Eigen::Index iz = (ik % nk12) % _nk(2);
        //        k = static_cast<double>(ix) / _nk(0) * _K.col(0) + static_cast<double>(iy) / _nk(1) * _K.col(1) + static_cast<double>(iz) / _nk(2) * _K.col(2);
        //    }
        //Eigen::MatrixBase<Derived>& k_ = const_cast<Eigen::MatrixBase<Derived>&>(k);
        //Eigen::VectorXd kfrac(m_K.cols());
        ArrayXindex idvec(m_K.cols());
        Eigen::Index nkv = m_nk.prod();
        for (Eigen::Index n = 0; n < m_K.cols(); ++n) {  // Disassemble flat index into index vector
            nkv /= m_nk(n);
            //kfrac(n) = static_cast<double>(ik / nkv) / m_nk(n) - 0.5;
            idvec(n) = id / nkv;
            id %= nkv;
        }
        //k_ = (m_K * kfrac.asDiagonal()).rowwise().sum();
        kVecAtIdVec(idvec, k);
    }
    
    void setMPIcomm(const MPI_Comm& comm);
    
    template <typename Derived>
    void primVecs(const Eigen::MatrixBase<Derived>& a);   // Set primative vectors
    const Eigen::MatrixXd& primVecs() const {return m_a;}   // Return primative vectors
    
    virtual void constructHamiltonian(const Eigen::VectorXd& k, Eigen::MatrixXcd& H) const;
    virtual void constructFermiVelocities(const int coord, const Eigen::VectorXd& k, Eigen::MatrixXcd& v) const;
    
    template <typename Derived, typename OtherDerived>
    void computeBands(const Eigen::DenseBase<Derived>& nk, const Eigen::DenseBase<OtherDerived>& kidpath);
    
    void computeDOS(const Eigen::Index nbins);
    
    void type(const std::string& tp) {m_type = tp;}   // Set type
    const std::string& type() const {return m_type;}   // Return type
    
    template <typename Derived>
    void dos(const std::array<double, 2>& erange, const Eigen::ArrayBase<Derived>& ds) {m_erange = erange; m_dos = ds;}  // Set DOS
    const Eigen::ArrayX2d& dos() const {return m_dos;}   // Return DOS
    
    const Eigen::ArrayXXd& bands() const {return m_bandpath;}
    
    const SqMatArray<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, 2>& hamDimerMag2d() const {return m_HdimerMag2d;}
    const SqMatArray2XXcd& fermiVdimerMag2d() const {return m_vdimerMag2d;}
    
    const std::array<double, 2>& energyRange() const {return m_erange;}
    
    const Eigen::MatrixXd& kPrimVecs() const {return m_K;}
    
    const ArrayXindex& kGridSizes() const {return m_nk;}
    
    template <typename Derived>
    void hopMatElem(const Eigen::DenseBase<Derived>& t_) {m_t = t_;}   // Set hoppong matrix elements
    std::complex<double> hopMatElem(const Eigen::Index i) const {return m_t(i);}  // Return hopping matrix element
    
    void chemPot(const double mu) {m_mu = mu;}   // Set chemical potential
    double chemPot() const {return m_mu;}   // Return chemical potential
    
    void moments(const SqMatArray22Xcd& mmts) {m_moments = mmts;}  // Set moments by copying
    void moments(SqMatArray22Xcd&& mmts) {m_moments = mmts;}  // Set moments by moving
    const SqMatArray22Xcd& moments() const {return m_moments;}   // Return moments
};


// Set unit cell vectors and record basic info
template <typename Derived>
void BareHamiltonian::primVecs(const Eigen::MatrixBase<Derived>& a) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument( "The dimension and number of primative vectors do not match!" );
    }
    else if (a.cols() == 0 || a.cols() > 3) {
        throw std::invalid_argument( "Primative vectors can only be of 1D, 2D, or 3D!" );
    }
    
    m_a = a;
    // Calculate reciprocal primative vectors
    m_K.resize(m_a.cols(), m_a.cols());
    if (m_a.cols() == 1) {
        m_v0 = std::fabs(m_a(0, 0));
        m_K(0, 0) = 2 * M_PI / m_a(0, 0);
    }
    else if (m_a.cols() == 2) {
        Eigen::Rotation2Dd rot90(M_PI / 2);
        Eigen::VectorXd tmp(2);
        tmp.noalias() = rot90 * m_a.col(1);
        m_v0 = std::fabs(m_a.col(0).dot(tmp));
        m_K.col(0) = (2 * M_PI) * tmp / m_a.col(0).dot(tmp);
        tmp.noalias() = rot90 * m_a.col(0);
        m_K.col(1) = (2 * M_PI) * tmp / m_a.col(1).dot(tmp);
    }
    else if (m_a.cols() == 3) {
        Eigen::Matrix3d ar = m_a;  // Cast dynamic-sized a to static 3*3 matrix to allow cross product
        m_v0 = ar.col(0).dot(ar.col(1).cross(ar.col(2)));
        m_K.col(0).noalias() = (2 * M_PI / m_v0) * ar.col(1).cross(ar.col(2));
        m_K.col(1).noalias() = (2 * M_PI / m_v0) * ar.col(2).cross(ar.col(0));
        m_K.col(2).noalias() = (2 * M_PI / m_v0) * ar.col(0).cross(ar.col(1));
        m_v0 = std::fabs(m_v0);
    }
}

template <typename Derived, typename OtherDerived>
void BareHamiltonian::computeBands(const Eigen::DenseBase<Derived>& nk, const Eigen::DenseBase<OtherDerived>& kidpath) {
    if (m_a.cols() != nk.size()) throw std::invalid_argument( "Space dimension of input k-point numbers did not match that of primative vectors!" );
    int is_inter;
    MPI_Comm_test_inter(m_comm, &is_inter);
    if (is_inter) throw std::invalid_argument( "MPI communicator is an intercommunicator prohibiting in-place Allreduce!" );
    
    Eigen::Index nbands;
    Eigen::VectorXd k = Eigen::VectorXd::Zero(m_a.rows());
    Eigen::MatrixXcd H;
    
    // Record info
    m_nk = nk;
    constructHamiltonian(k, H);   // Call the implemented method in derived classes; read out H at a k-vector once to get basic info
    nbands = H.rows();
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es0(nbands);
    
    // Calculate bands
    const Eigen::Index nkt = m_nk.prod();
    Eigen::Index kid;
    mostEvenPart(nkt, m_psize, m_prank, m_klocalsize, m_klocalstart);
    m_bands.resize(nbands + k.size(), nkt);   // Allocate full-size data; first several rows is k-vector
    for (Eigen::Index kidlocal = 0; kidlocal < m_klocalsize; ++kidlocal) {
        kid = kidlocal + m_klocalstart;
        kVecAtFlatId(kid, k);
        constructHamiltonian(k, H);  // Call the implemented method in derived classes
        es0.compute(H, Eigen::EigenvaluesOnly);  // Only the lower triangular part is used
        m_bands.block(0, kid, k.size(), 1) = k;   // Record k-vectors
        m_bands.block(k.size(), kid, nbands, 1) = es0.eigenvalues();   // Record bands
    }
    // All-gather bands data
    int *counts = new int[m_psize];
    int *displs = new int[m_psize];
    counts[m_prank] = m_bands.rows() * m_klocalsize;
    displs[m_prank] = m_bands.rows() * m_klocalstart;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, counts, 1, MPI_INT, m_comm);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, displs, 1, MPI_INT, m_comm);
    MPI_Allgatherv(MPI_IN_PLACE, counts[m_prank], MPI_DOUBLE, m_bands.data(), counts, displs, MPI_DOUBLE, m_comm);
    delete[] counts;
    delete[] displs;
    
    // Record info because finding max or min is a little bit costly
    //if (m_klocalsize > 0) {
    //    m_erange[0] = m_bands.bottomRows(nbands).minCoeff();
    //    m_erange[1] = m_bands.bottomRows(nbands).maxCoeff();
    //}
    //else {
    //    m_erange[0] = 1e9 * m_t.abs().maxCoeff();
    //    m_erange[1] = -m_erange[0];
    //}
    //MPI_Allreduce(MPI_IN_PLACE, &m_erange[0], 1, MPI_DOUBLE, MPI_MIN, m_comm);
    //MPI_Allreduce(MPI_IN_PLACE, &m_erange[1], 1, MPI_DOUBLE, MPI_MAX, m_comm);
    m_erange[0] = m_bands.bottomRows(nbands).minCoeff();
    m_erange[1] = m_bands.bottomRows(nbands).maxCoeff();
    
    // Fill bands along paths
    m_bandpath.resize(m_bands.rows(), kidpath.cols());
    for (Eigen::Index ik = 0; ik < kidpath.cols(); ++ik) m_bandpath.col(ik) = m_bands.col(flatIndex(kidpath.col(ik)));
    //m_bandpath(Eigen::seq(m_K.rows(), Eigen::last), Eigen::all) -= m_mu;
    
    // Calculate the block diagonal Hamiltonian for the special case of 2D dimer Hubbard model in magnetic field
    if (m_type == "dimer_mag_2d") {
        if (m_a.cols() != 2) throw std::invalid_argument("Space dimension must be 2 for 2D dimer Hubbard model in magnetic fields!");
        if (nbands % 2 != 0) throw std::range_error("Hamiltonian's dimension must be multiple of 2 for 2D dimer Hubbard model in magnetic fields!");
        const Eigen::Index nb_2 = nbands / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(nb_2);
        Eigen::MatrixXcd fv;
        m_HdimerMag2d.mpiComm(m_comm);
        m_HdimerMag2d.resize(nkt, nb_2, 2);  // Allocate resources for _HdimerMag2d, full-size data is needed by all processes
        auto Hmastpart = m_HdimerMag2d.mastDim0Part();  // Choose only partitioning the first dimension
        m_vdimerMag2d.resize(2, Hmastpart.dim0(), nb_2);   // No need to gather, so just allocate local-size data
        Eigen::Index m;
        int co;
        for (Eigen::Index kidlocal = 0; kidlocal < Hmastpart.dim0(); ++kidlocal) {
            kVecAtFlatId(kidlocal + Hmastpart.displ(), k);
            constructHamiltonian(k, H);  // Call the implemented method in derived classes
            // Block diagonalize w.r.t. the magnetic unit cell. The key is that the top-left and bottom-right blocks are Hermitian
            // (and they are the same) and that the top-right and bottom-left blocks are already identity matrices multiplied by a
            // constant. Therefore we can diagonalize the four blocks by a unitary matrix.
            es.compute(H.topLeftCorner(nb_2, nb_2));  // Only the lower triangular part is used
            // Assemble the block Hamiltonian in the dimer space
            for(m = 0; m < nb_2; ++m) Hmastpart(kidlocal, m) << es.eigenvalues()(m), std::conj(H(nb_2, 0)),
                                                                H(nb_2, 0),          es.eigenvalues()(m);
            // Caculate Fermi velocity matrices. fv is block diagonal and the top left and bottom right blocks are the same.
            for (co = 0; co < 2; ++co) {
                constructFermiVelocities(co, k, fv);
                m_vdimerMag2d(co, kidlocal).noalias() = es.eigenvectors().adjoint() * fv.topLeftCorner(nb_2, nb_2).selfadjointView<Eigen::Lower>() * es.eigenvectors();
            }
        }
        Hmastpart.allGather();   // All gather because all processes need the full data
    }
}








// Function templates should better be implemented in the same file in which it is defined, or need to use some other method
// to deal with the linking issue
template <typename Scalar, int n1, int nm>
void simpsonIntegrate(const SqMatArray<Scalar, 1, n1, nm>& integrand, const double dx, Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > result) {  // Eigen::Ref<Eigen::MatrixXcd> also references fixed size matrix
    if (integrand.dim1() % 2 == 0) throw std::range_error("#grid size must be odd for Simpson integration!");
    result.resize(integrand.dimm(), integrand.dimm());
    result.setZero();
    for (Eigen::Index i = 0; i + 2 < integrand.dim1(); i += 2) result += (dx / 3.0) * (integrand[i] + 4.0 * integrand[i + 1] + integrand[i + 2]);
}

template <typename Derived>
typename Derived::Scalar simpsonIntegrate(const Eigen::DenseBase<Derived>& integrand, const double dx) {
    if (integrand.size() % 2 == 0) throw std::range_error("#grid size must be odd for Simpson integration!");
    typename Derived::Scalar result = 0.0;
    for (Eigen::Index i = 0; i + 2 < integrand.size(); i += 2) result += (dx / 3.0) * (integrand(i) + 4.0 * integrand(i + 1) + integrand(i + 2));
    return result;
}

template <int n0, int n1, int nm, typename Derived>
void computeLattGFfCoeffs(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen_dyn,
                          const SqMatArray<std::complex<double>, n0, 1, nm>& selfen_static, const Eigen::DenseBase<Derived>& energies,
                          SqMatArray<std::complex<double>, n0, n1, nm>& Gw) {
    const Eigen::Index nc = selfen_dyn.dimm();
    Gw.mpiComm(selfen_dyn.mpiComm());
    Gw.resize(selfen_dyn.dim0(), selfen_dyn.dim1(), nc);
    assert(selfen_dyn.dim1() == energies.size());
    const auto selfen_dyn_mastpart = selfen_dyn.mastFlatPart();
    auto Gwmastpart = Gw.mastFlatPart();
    std::array<Eigen::Index, 2> so;
    std::complex<double> wu;
    
    if (nc == 1) {  // Single site case, where we can utilize the noninteracting density of states
        const Eigen::Index nbins = H0.dos().rows();
        if (nbins == 0) throw std::range_error("DOS has not been computed or set!");
        Eigen::Index ie;
        const double binsize = (H0.energyRange()[1] - H0.energyRange()[0]) / nbins;
        Eigen::ArrayXcd integrand(nbins);
        
        for (Eigen::Index i = 0; i < selfen_dyn_mastpart.size(); ++i) {
            so = selfen_dyn_mastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
        //    if constexpr (std::is_same<typename Derived::Scalar, std::complex<mpfr::mpreal> >::value) {
        //        wu.real(energies(so[1]).real().toDouble() + H0.chemPot());
        //        wu.imag(energies(so[1]).imag().toDouble());
        //    }
        //    else wu = static_cast<std::complex<double> >(energies(so[1])) + H0.chemPot();
            wu = energies(so[1]) + H0.chemPot();
            // Glat.masteredPart(i).setZero();
            // Update the lattice Green's function
            for (ie = 0; ie < nbins; ++ie) {
                integrand(ie) = -H0.dos()(ie, 1) / (wu - H0.dos()(ie, 0) - selfen_dyn_mastpart(i, 0, 0) - selfen_static(so[0], 0, 0, 0));
                // Glat.masteredPart(i).noalias() -= binsize * _H0->dos()(ie) * ((1i * w + _H0->mu - e) * Eigen::MatrixXcd::Identity(nc, nc) - selfenergy.masteredPart(i)).inverse();
            }
            Gwmastpart(i, 0, 0) = simpsonIntegrate(integrand, binsize);
            // Add head and tail parts to the integral
            Gwmastpart(i, 0, 0) += (binsize / 4) * (integrand(0) + integrand(nbins - 1));
        }
    }
    else if (nc > 1) {
        const Eigen::Index spatiodim = H0.kPrimVecs().rows();
        if (spatiodim == 0) throw std::range_error("Reciprocal primative vectors have not been set!");
        // else if (spatodim > 2) throw std::range_error("k-space integration has only been implemented for 1- and 2-dimensional cases!");
        // SqMatArray<std::complex<double>, 1, Eigen::Dynamic, Eigen::Dynamic> integrand0(nk, nc);
        
        if (H0.type() == "dimer_mag_2d") {
            if (H0.hamDimerMag2d().size() == 0) throw std::range_error("Block Hamiltonian of the 2D dimer Hubbard model in magnetic fields has not been computed!");
            Eigen::Index ist;
            Gwmastpart().setZero();
            for (Eigen::Index i = 0; i < selfen_dyn_mastpart.size(); ++i) {
                so = selfen_dyn_mastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
            //    if constexpr (std::is_same<typename Derived::Scalar, std::complex<mpfr::mpreal> >::value) {
            //        wu.real(energies(so[1]).real().toDouble() + H0.chemPot());
            //        wu.imag(energies(so[1]).imag().toDouble());
            //    }
            //    else wu = static_cast<std::complex<double> >(energies(so[1])) + H0.chemPot();
                wu = energies(so[1]) + H0.chemPot();
                for (ist = 0; ist < H0.hamDimerMag2d().size(); ++ist) Gwmastpart[i] += -(wu * Eigen::Matrix2cd::Identity() - H0.hamDimerMag2d()[ist]
                                                                                         - selfen_dyn_mastpart[i] - selfen_static[so[0]]).inverse();
                Gwmastpart[i] /= static_cast<double>(H0.hamDimerMag2d().size());
            }
        }
        else {
            if (H0.kGridSizes().size() == 0) throw std::range_error("Grid size for the reciprocal space has not been set!");
            const Eigen::Index nk = H0.kGridSizes().prod();
            Eigen::Index ik;
            Eigen::VectorXd k;
            Eigen::MatrixXcd H;
            Gwmastpart().setZero();
            for (Eigen::Index i = 0; i < selfen_dyn_mastpart.size(); ++i) {
                so = selfen_dyn_mastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
            //    if constexpr (std::is_same<typename Derived::Scalar, std::complex<mpfr::mpreal> >::value) {
            //        wu.real(energies(so[1]).real().toDouble() + H0.chemPot());
            //        wu.imag(energies(so[1]).imag().toDouble());
            //    }
            //    else wu = static_cast<std::complex<double> >(energies(so[1])) + H0.chemPot();
                wu = energies(so[1]) + H0.chemPot();
                for (ik = 0; ik < nk; ++ik) {
                    H0.kVecAtFlatId(ik, k);
                    H0.constructHamiltonian(k, H);
                    Gwmastpart[i] += -(wu * Eigen::MatrixXcd::Identity(nc, nc) - H.selfadjointView<Eigen::Lower>() * Eigen::MatrixXcd::Identity(nc, nc) - selfen_dyn_mastpart[i] - selfen_static[so[0]]).inverse();
                }
                Gwmastpart[i] /= static_cast<double>(nk);
            }
        }
    }
}

template <int n0, int n1, int nm, typename Derived>
void computeSpectraW(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen,
                     const Eigen::DenseBase<Derived>& energies, SqMatArray<std::complex<double>, n0, n1, nm>& Aw) {
    SqMatArray<std::complex<double>, n0, 1, nm> selfenstatic(selfen.dim0(), 1, selfen.dimm());
    selfenstatic().setZero();
    computeLattGFfCoeffs(H0, selfen, selfenstatic, energies, Aw);
    auto Awpart = Aw.mastFlatPart();
    for (Eigen::Index i = 0; i < Awpart.size(); ++i) Awpart[i] = (Awpart[i] - Awpart[i].adjoint().eval()) / (2i * M_PI);
    Awpart.allGather();
}

template <int n0, int n1, int nm>
void computeSpectraKW0(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen, const Eigen::Index id0,
                       SqMatArray<std::complex<double>, n0, n1, nm>& A0) {
    A0.mpiComm(selfen.mpiComm());
    const Eigen::Index nk = H0.kGridSizes().prod();
    A0.resize(selfen.dim0(), nk, selfen.dimm());
    auto A0part = A0.mastFlatPart();
    std::array<Eigen::Index, 2> sk;
    
    if (H0.type() == "dimer_mag_2d") {
        for (Eigen::Index i = 0; i < A0part.size(); ++i) {
            sk = A0part.global2dIndex(i);
            A0part[i].setZero();
            for (Eigen::Index m = 0; m < H0.hamDimerMag2d().dim1(); ++m)
                A0part[i] += -(H0.chemPot() * Eigen::Matrix2cd::Identity() - H0.hamDimerMag2d()(sk[1], m) - selfen(sk[0], id0)).inverse();
            A0part[i] = (A0part[i] - A0part[i].adjoint().eval()) / (static_cast<double>(H0.hamDimerMag2d().dim1()) * 2i * M_PI);
        }
    }
    else {
        Eigen::VectorXd k;
        Eigen::MatrixXcd H;
        for (Eigen::Index i = 0; i < A0part.size(); ++i) {
            sk = A0part.global2dIndex(i);
            H0.kVecAtFlatId(sk[1], k);
            H0.constructHamiltonian(k, H);
            A0part[i] = -(H0.chemPot() * Eigen::MatrixXcd::Identity(selfen.dimm(), selfen.dimm())
                          - H.selfadjointView<Eigen::Lower>() * Eigen::MatrixXcd::Identity(selfen.dimm(), selfen.dimm()) - selfen(sk[0], id0)).inverse();
            A0part[i] = (A0part[i] - A0part[i].adjoint().eval()) / (2i * M_PI);
        }
    }
    A0part.allGather();
}

// n1 dimension of A first runs over energy and then k points; kidpath: each column stores index vector of k point
template <int n0, int n1, int nm, typename Derived, typename OtherDerived>
void computeSpectraKW(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen, const Eigen::DenseBase<Derived>& energies,
                      const Eigen::DenseBase<OtherDerived>& kidpath, SqMatArray<std::complex<double>, n0, n1, nm>& Akw) {
    Akw.mpiComm(selfen.mpiComm());
    Akw.resize(selfen.dim0(), energies.size() * kidpath.cols(), selfen.dimm());
    auto Akwpart = Akw.mastFlatPart();
    std::array<Eigen::Index, 2> swk;
    Eigen::Index iw, ik;
    
    if (H0.type() == "dimer_mag_2d") {
        Eigen::Index kid;
        for (Eigen::Index i = 0; i < Akwpart.size(); ++i) {
            swk = Akwpart.global2dIndex(i);
            iw = swk[1] % energies.size();
            ik = swk[1] / energies.size();
            kid = H0.flatIndex(kidpath.col(ik));
            Akwpart[i].setZero();
            for (Eigen::Index m = 0; m < H0.hamDimerMag2d().dim1(); ++m)
                Akwpart[i] += -((energies(iw) + H0.chemPot()) * Eigen::Matrix2cd::Identity() - H0.hamDimerMag2d()(kid, m) - selfen(swk[0], iw)).inverse();
            Akwpart[i] = (Akwpart[i] - Akwpart[i].adjoint().eval()) / (static_cast<double>(H0.hamDimerMag2d().dim1()) * 2i * M_PI);
        }
    }
    else {
        Eigen::VectorXd k;
        Eigen::MatrixXcd H;
        for (Eigen::Index i = 0; i < Akwpart.size(); ++i) {
            swk = Akwpart.global2dIndex(i);
            iw = swk[1] % energies.size();
            ik = swk[1] / energies.size();
            H0.kVecAtIdVec(kidpath.col(ik), k);
            H0.constructHamiltonian(k, H);
            Akwpart[i] = -((energies(iw) + H0.chemPot()) * Eigen::MatrixXcd::Identity(selfen.dimm(), selfen.dimm())
                          - H.selfadjointView<Eigen::Lower>() * Eigen::MatrixXcd::Identity(selfen.dimm(), selfen.dimm()) - selfen(swk[0], iw)).inverse();
            Akwpart[i] = (Akwpart[i] - Akwpart[i].adjoint().eval()) / (2i * M_PI);
        }
    }
    Akwpart.allGather();
}


#endif /* bare_hamiltonian_hpp */
