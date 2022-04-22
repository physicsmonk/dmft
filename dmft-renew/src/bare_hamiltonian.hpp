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
#include "gf_data_wrapper.hpp"



typedef Eigen::Array<std::size_t, Eigen::Dynamic, 1> ArrayXsizet;

// Base class for user-defined bare Hamiltonians. Users want to implement the virtual method
// constructHamiltonian in derived classes to construct any Hamiltonian matrix they want.
class BareHamiltonian {
private:
    MPI_Comm _comm;
    int _psize, _prank;
    std::size_t _kbbsize, _klocalsize, _klocalstart;
    double _v0;   //  Unit cell volume/area/length
    Eigen::MatrixXd _K;  // Stores reciprocal primative vectors in columns
    ArrayXsizet _nk;   // Numbers of k-points along each reciprocal primative vector
    std::array<double, 2> _erange;   // Energy range of the band structure
    Eigen::ArrayXXd _bands;  // Stores energy bands; index is of (energy, (kx, ky, kz))
    // Block diagonalized Hamiltonian for the special case, 2D dimer Hubbard model in magnetic fields. First index runs over k-vectors (ky major)
    // and the second index runs over the eigenvalue space of the block diagonalization.
    SqMatArray<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, 2> _HdimerMag2d;
    // Fermi velocity matrices in the space of _HdimerMag2d for the special case, 2D dimer Hubbard model in magnetic fields.
    // Local-size: first index is just x or y, second index runs over local k-vectors (ky major)
    SqMatArray2XXcd _vdimerMag2d;
    std::string _type;
    Eigen::ArrayXd _dos;   // Stores density of states
    double _mu;  // Chemical potential. Note band structure and DOS are independent of it.
    SqMatArray21Xcd _m1, _m2;  // First and second moments
    
protected:
    Eigen::MatrixXd a;  // Stores primative vectors in columns
    Eigen::ArrayXcd t;  // Stores hopping matrix elements
    
public:
    // No need for an explicit constructor, which is also user-friendly
    virtual ~BareHamiltonian() {}
    
    // Provide a method for index converting for k-space storage, otherwise we could make a data wrapper
    std::size_t flatIndex(const std::size_t ix, const std::size_t iy, const std::size_t iz) const {
        if (a.cols() != 3) throw std::bad_function_call();
        return (ix * _nk(1) + iy) * _nk(2) + iz;
    }
    std::size_t flatIndex(const std::size_t ix, const std::size_t iy) const {
        if (a.cols() != 2) throw std::bad_function_call();
        return ix * _nk(1) + iy;
    }
    
    void kVecAtIndex(std::size_t ik, Eigen::VectorXd& k) const;  // Calculate the ik-th k vector
    
    void setMPIcomm(const MPI_Comm& comm);
    
    template <typename Derived>
    void primVecs(const Eigen::MatrixBase<Derived>& a_);   // Set primative vectors
    const Eigen::MatrixXd& primVecs() const {return a;}   // Return primative vectors
    
    virtual void constructHamiltonian(const Eigen::VectorXd& k, Eigen::MatrixXcd& H) const;
    virtual void constructFermiVelocities(const int coord, const Eigen::VectorXd& k, Eigen::MatrixXcd& v) const;
    
    template <typename Derived>
    void computeBands(const Eigen::DenseBase<Derived>& nk);
    
    void computeDOS(const std::size_t nbins);
    
    void type(const std::string& tp) {_type = tp;}   // Set type
    const std::string& type() const {return _type;}   // Return type
    
    template <typename Derived>
    void dos(const std::array<double, 2>& erange, const Eigen::ArrayBase<Derived>& ds) {_erange = erange; _dos = ds;}  // Set DOS
    const Eigen::ArrayXd& dos() const {return _dos;}   // Return DOS
    
    const Eigen::ArrayXXd& bands() const {return _bands;}
    
    const SqMatArray<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, 2>& hamDimerMag2d() const {return _HdimerMag2d;}
    const SqMatArray2XXcd& fermiVdimerMag2d() const {return _vdimerMag2d;}
    
    const std::array<double, 2>& energyRange() const {return _erange;}
    
    const Eigen::MatrixXd& kPrimVecs() const {return _K;}
    
    const ArrayXsizet& kGridSizes() const {return _nk;}
    
    template <typename Derived>
    void hopMatElem(const Eigen::DenseBase<Derived>& t_) {t = t_;}   // Set hoppong matrix elements
    std::complex<double> hopMatElem(const std::size_t i) const {return t(i);}  // Return hopping matrix element
    
    void chemPot(const double mu) {_mu = mu;}   // Set chemical potential
    double chemPot() const {return _mu;}   // Return chemical potential
    
    void firstMoment(const SqMatArray21Xcd& m1) {_m1 = m1;}  // Set first moment
    const SqMatArray21Xcd& firstMoment() const {return _m1;}   // Return first moment
    void secondMoment(const SqMatArray21Xcd& m2) {_m2 = m2;}  // Set second moment
    const SqMatArray21Xcd& secondMoment() const {return _m2;}   // Return second moment
};

// Set unit cell vectors and record basic info
template <typename Derived>
void BareHamiltonian::primVecs(const Eigen::MatrixBase<Derived>& a_) {
    if (a_.rows() != a_.cols()) {
        throw std::invalid_argument( "The dimension and number of primative vectors do not match!" );
    }
    else if (a_.cols() == 0 || a_.cols() > 3) {
        throw std::invalid_argument( "Primative vectors can only be of 1D, 2D, or 3D!" );
    }
    
    a = a_;
    // Calculate reciprocal primative vectors
    _K.resize(a.cols(), a.cols());
    if (a.cols() == 1) {
        _v0 = std::fabs(a(0, 0));
        _K(0, 0) = 2 * M_PI / a(0, 0);
    }
    else if (a.cols() == 2) {
        Eigen::Rotation2Dd rot90(M_PI / 2);
        Eigen::VectorXd tmp(2);
        tmp.noalias() = rot90 * a.col(1);
        _v0 = std::fabs(a.col(0).dot(tmp));
        _K.col(0) = (2 * M_PI) * tmp / a.col(0).dot(tmp);
        tmp.noalias() = rot90 * a.col(0);
        _K.col(1) = (2 * M_PI) * tmp / a.col(1).dot(tmp);
    }
    else if (a.cols() == 3) {
        Eigen::Matrix3d ar = a;  // Cast dynamic-sized a to static 3*3 matrix to allow cross product
        _v0 = ar.col(0).dot(ar.col(1).cross(ar.col(2)));
        _K.col(0).noalias() = (2 * M_PI / _v0) * ar.col(1).cross(ar.col(2));
        _K.col(1).noalias() = (2 * M_PI / _v0) * ar.col(2).cross(ar.col(0));
        _K.col(2).noalias() = (2 * M_PI / _v0) * ar.col(0).cross(ar.col(1));
        _v0 = std::fabs(_v0);
    }
}

template <typename Derived>
void BareHamiltonian::computeBands(const Eigen::DenseBase<Derived>& nk) {
    if (a.cols() != nk.size()) throw std::invalid_argument( "Space dimension of input k-point numbers did not match that of primative vectors!" );
    int is_inter;
    MPI_Comm_test_inter(_comm, &is_inter);
    if (is_inter) throw std::invalid_argument( "MPI communicator is an intercommunicator prohibiting in-place Allreduce!" );
    
    std::size_t nbands, ik;
    Eigen::VectorXd k = Eigen::VectorXd::Zero(a.rows());
    Eigen::MatrixXcd H;
    
    // Record info
    _nk = nk;
    constructHamiltonian(k, H);   // Call the implemented method in derived classes; read out H at a k-vector once to get basic info
    nbands = H.rows();
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es0(nbands);
    
    // Calculate bands
    const std::size_t nkt = _nk.prod();
    _kbbsize = nkt / _psize;
    _klocalsize = (_prank < _psize - 1) ? _kbbsize : _kbbsize + nkt % _psize;
    _klocalstart = _prank * _kbbsize;
    _bands.resize(nbands, _klocalsize);   // Allocate local-sized _bands because it is not directly used in the program
    for (ik = 0; ik < _klocalsize; ++ik) {
        kVecAtIndex(ik + _klocalstart, k);
        constructHamiltonian(k, H);  // Call the implemented method in derived classes
        es0.compute(H, Eigen::EigenvaluesOnly);  // Only the lower triangular part is used
        _bands.col(ik) = es0.eigenvalues();
    }
    
    // Record info because finding max or min is a little bit costly
    _erange[0] = _bands.minCoeff();
    _erange[1] = _bands.maxCoeff();
    MPI_Allreduce(MPI_IN_PLACE, &_erange[0], 1, MPI_DOUBLE, MPI_MIN, _comm);
    MPI_Allreduce(MPI_IN_PLACE, &_erange[1], 1, MPI_DOUBLE, MPI_MAX, _comm);
    
    // Calculate the block diagonal Hamiltonian for the special case of 2D dimer Hubbard model in magnetic field
    if (_type == "dimer_mag_2d") {
        if (a.cols() != 2) throw std::invalid_argument("Space dimension must be 2 for 2D dimer Hubbard model in magnetic fields!");
        if (nbands % 2 != 0) throw std::range_error("Hamiltonian's dimension must be multiple of 2 for 2D dimer Hubbard model in magnetic fields!");
        const std::size_t nb_2 = nbands / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(nb_2);
        Eigen::MatrixXcd fv;
        _HdimerMag2d.mpiCommunicator(_comm);
        _HdimerMag2d.resize(nkt, nb_2, 2);  // Allocate resources for _HdimerMag2d, full-size data is needed by all processes
        auto Hmastpart = _HdimerMag2d.mastDim0Part();  // Choose only partitioning the first dimension
        _vdimerMag2d.resize(2, Hmastpart.dim0(), nb_2);   // No need to gather, so just allocate local-size data
        std::size_t m;
        int co;
        for (ik = 0; ik < Hmastpart.dim0(); ++ik) {
            kVecAtIndex(ik + Hmastpart.start(), k);
            constructHamiltonian(k, H);  // Call the implemented method in derived classes
            // Block diagonalize w.r.t. the magnetic unit cell. The key is that the top-left and bottom-right blocks are Hermitian
            // (and they are the same) and that the top-right and bottom-left blocks are already identity matrices multiplied by a
            // constant. Therefore we can diagonalize the four blocks by a unitary matrix.
            es.compute(H.topLeftCorner(nb_2, nb_2));  // Only the lower triangular part is used
            // Assemble the block Hamiltonian in the dimer space
            for(m = 0; m < nb_2; ++m) Hmastpart(ik, m) << es.eigenvalues()(m), std::conj(H(nb_2, 0)),
                                                          H(nb_2, 0),          es.eigenvalues()(m);
            // Caculate Fermi velocity matrices. fv is block diagonal and the top left and bottom right blocks are the same.
            for (co = 0; co < 2; ++co) {
                constructFermiVelocities(co, k, fv);
                _vdimerMag2d(co, ik).noalias() = es.eigenvectors().adjoint() * fv.topLeftCorner(nb_2, nb_2).selfadjointView<Eigen::Lower>() * es.eigenvectors();
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
    for (std::size_t i = 0; i + 2 < integrand.dim1(); i += 2) result += (dx / 3.0) * (integrand[i] + 4.0 * integrand[i + 1] + integrand[i + 2]);
}

template <typename Derived>
typename Derived::Scalar simpsonIntegrate(const Eigen::DenseBase<Derived>& integrand, const double dx) {
    if (integrand.size() % 2 == 0) throw std::range_error("#grid size must be odd for Simpson integration!");
    typename Derived::Scalar result = 0.0;
    for (std::size_t i = 0; i + 2 < integrand.size(); i += 2) result += (dx / 3.0) * (integrand(i) + 4.0 * integrand(i + 1) + integrand(i + 2));
    return result;
}

// Requires that Gw has been properly allocated, i.e., has the same shape and mpi partition as selfen
template <int n0, int n1, int nm, typename Derived, int othern0, int othern1, int othernm>
void computeLattGFfCoeffs(const BareHamiltonian& H0, const SqMatArray<std::complex<double>, n0, n1, nm>& selfen, const Eigen::DenseBase<Derived>& energies, SqMatArray<std::complex<double>, othern0, othern1, othernm>& Gw) {
    const std::size_t nc = selfen.dimm();
    assert(selfen.dim0() == Gw.dim0() && selfen.dim1() == energies.size() && selfen.dim1() == Gw.dim1() && nc == Gw.dimm());
    const auto selfenmastpart = selfen.mastFlatPart();
    auto Gwmastpart = Gw.mastFlatPart();
    std::array<std::size_t, 2> so;
    std::complex<double> wu;
    
    if (nc == 1) {  // Single site case, where we can utilize the noninteracting density of states
        const std::size_t nbins = H0.dos().size();
        if (nbins == 0) throw std::range_error("DOS has not been computed or set!");
        std::size_t ie;
        const double binsize = (H0.energyRange()[1] - H0.energyRange()[0]) / nbins;
        double e;
        Eigen::ArrayXcd integrand(nbins);
        
        for (std::size_t i = 0; i < selfenmastpart.size(); ++i) {
            so = selfenmastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
            wu = static_cast<std::complex<double> >(energies(so[1])) + H0.chemPot();
            // Glat.masteredPart(i).setZero();
            // Update the lattice Green's function
            for (ie = 0; ie < nbins; ++ie) {
                e = (ie + 0.5) * binsize + H0.energyRange()[0];
                integrand(ie) = -H0.dos()(ie) / (wu - e - selfenmastpart(i, 0, 0));
                // Glat.masteredPart(i).noalias() -= binsize * _H0->dos()(ie) * ((1i * w + _H0->mu - e) * Eigen::MatrixXcd::Identity(nc, nc) - selfenergy.masteredPart(i)).inverse();
            }
            Gwmastpart(i, 0, 0) = simpsonIntegrate(integrand, binsize);
            // Add head and tail parts to the integral
            Gwmastpart(i, 0, 0) += (binsize / 4) * (integrand(0) + integrand(nbins - 1));
        }
    }
    else if (nc > 1) {
        const std::size_t spatodim = H0.kPrimVecs().rows();
        if (spatodim == 0) throw std::range_error("Reciprocal primative vectors have not been set!");
        // else if (spatodim > 2) throw std::range_error("k-space integration has only been implemented for 1- and 2-dimensional cases!");
        // SqMatArray<std::complex<double>, 1, Eigen::Dynamic, Eigen::Dynamic> integrand0(nk, nc);
        
        if (H0.type() == "dimer_mag_2d") {
            if (H0.hamDimerMag2d().size() == 0) throw std::range_error("Block Hamiltonian of the 2D dimer Hubbard model in magnetic fields has not been computed!");
            std::size_t ist;
            Gwmastpart().setZero();
            for (std::size_t i = 0; i < selfenmastpart.size(); ++i) {
                so = selfenmastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
                wu = static_cast<std::complex<double> >(energies(so[1])) + H0.chemPot();
                for (ist = 0; ist < H0.hamDimerMag2d().size(); ++ist) {
                    Gwmastpart[i] += -(wu * Eigen::Matrix2cd::Identity() - H0.hamDimerMag2d()[ist] - selfenmastpart[i]).inverse();
                }
                Gwmastpart[i] /= static_cast<double>(H0.hamDimerMag2d().size());
            }
        }
        else {
            if (H0.kGridSizes().size() == 0) throw std::range_error("Grid size for the reciprocal space has not been set!");
            const std::size_t nk = H0.kGridSizes().prod();
            std::size_t ik;
            Eigen::VectorXd k;
            Eigen::MatrixXcd H;
            Gwmastpart().setZero();
            for (std::size_t i = 0; i < selfenmastpart.size(); ++i) {
                so = selfenmastpart.global2dIndex(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
                wu = static_cast<std::complex<double> >(energies(so[1])) + H0.chemPot();
                for (ik = 0; ik < nk; ++ik) {
                    H0.kVecAtIndex(ik, k);
                    H0.constructHamiltonian(k, H);
                    Gwmastpart[i] += -(wu * Eigen::MatrixXcd::Identity(nc, nc) - H.selfadjointView<Eigen::Lower>() * Eigen::MatrixXcd::Identity(nc, nc) - selfenmastpart[i]).inverse();
                }
                Gwmastpart[i] /= static_cast<double>(nk);
            }
        }
    }
}


#endif /* bare_hamiltonian_hpp */
