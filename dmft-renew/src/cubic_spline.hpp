//
//  cubic_spline.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef cubic_spline_hpp
#define cubic_spline_hpp

#include <complex>
#include <cmath>
#include <algorithm>
#include <functional>   // std::reference_wrapper
#include <Eigen/LU>
#include "gf_data_wrapper.hpp"

using namespace std::complex_literals;


// We choose to link to external x and y data, not copy them, because we have them stored them in other places anyway
template <typename _ScalarT, int _n0, int _n1, int _nm>
class CubicSplineMat {
private:
    static constexpr int m_sc_nmatelems = _n0 == Eigen::Dynamic || _nm == Eigen::Dynamic ? Eigen::Dynamic : _n0 * _nm * _nm;
    double m_dx;  // Store interval of x coordinates for equidistant case
    const Eigen::Array<double, _n1, 1> *m_ptr2x;   // A link the const external x data
    // Each Column stores the second derivative for data points of each matrix element. Rows are in (m_col, m_row, dim0)-major order.
    Eigen::Matrix<_ScalarT, _n1, m_sc_nmatelems> m_deriv2;
    const SqMatArray<_ScalarT, _n0, _n1, _nm> *m_ptr2y;  // A link to the const external y data
    
    std::size_t flatIndex(const std::size_t i0, const std::size_t im0, const std::size_t im1) const {
        assert(i0 < m_ptr2y->dim0() && im0 < m_ptr2y->dimm() && im1 < m_ptr2y->dimm());
        return (i0 * m_ptr2y->dimm() + im1) * m_ptr2y->dimm() + im0;
    }
    
public:
    // Special boundary conditions are y'(x_min) + y'(x_max) = G1 and y"(x_min) + y"(x_max) = G2; assumes x is already sorted in ascending order.
    // Pass pointers to x and y not const references because x and y will be accessed later on so we want passing const temporaries trigger compliation error
    void build(const Eigen::Array<double, _n1, 1> *x, const SqMatArray<_ScalarT, _n0, _n1, _nm> *y, const SqMatArray<_ScalarT, _n0, 2, _nm>& Ghfc);
    
    CubicSplineMat() : m_ptr2x(nullptr), m_ptr2y(nullptr), m_dx(0.0) {}
    // Build spline on construction
    CubicSplineMat(const Eigen::Array<double, _n1, 1> *x, const SqMatArray<_ScalarT, _n0, _n1, _nm> *y, const SqMatArray<_ScalarT, _n0, 2, _nm>& Ghfc) : m_ptr2x(x), m_ptr2y(y), m_dx((*x)(1) - (*x)(0)) {build(x, y, Ghfc);}
    CubicSplineMat(const CubicSplineMat&) = default;
    CubicSplineMat(CubicSplineMat&&) = default;
    CubicSplineMat& operator=(const CubicSplineMat&) = default;
    CubicSplineMat& operator=(CubicSplineMat&&) = default;
    ~CubicSplineMat() = default;
    
    _ScalarT operator() (const std::size_t i0, const double x0, const std::size_t im0, const std::size_t im1) const;
    // Equidistant version for getting a y value at arbitrary x. Specialize this for efficiency.
    _ScalarT equidistAt(const std::size_t i0, const double x0, const std::size_t im0, const std::size_t im1) const;
    
    std::complex<double> fourierTransform(const std::size_t i0, const double omega, const std::size_t im0, const std::size_t im1) const;
    
    void fourierTransform(const std::size_t i0, const double omega, Eigen::Ref<Eigen::Matrix<_ScalarT, Eigen::Dynamic, Eigen::Dynamic> > result) const;
};

// This building algorithm is in terms of the second derivative. See in this link en.wikiversity.org/wiki/Cubic_Spline_Interpolation
template <typename _ScalarT, int _n0, int _n1, int _nm>
void CubicSplineMat<_ScalarT, _n0, _n1, _nm>::build(const Eigen::Array<double, _n1, 1> *x, const SqMatArray<_ScalarT, _n0, _n1, _nm> *y, const SqMatArray<_ScalarT, _n0, 2, _nm>& Ghfc) {
    static_assert(std::is_same<_ScalarT, double>::value || std::is_same<_ScalarT, std::complex<double> >::value);
    assert(x->size() == y->dim1() && x->size() > 1);
    
    // Link to external data
    m_ptr2x = x;
    m_ptr2y = y;
    m_dx = (*x)(1) - (*x)(0);
    
    // Get run-time dimmension
    const std::size_t np = x->size();
    
    // Matrix for the linear equations, always real because x is real; automatically asserts _n1 == np for fixed-size case
    // At the beginning fill the diagonal part
    Eigen::Matrix<double, _n1, _n1> A = 2.0 * Eigen::Matrix<double, _n1, _n1>::Identity(np, np);
    // Matrix for the rhs values, automatically asserts _n_mat_elems == _dim0 * nmnm for fixed-size case
    Eigen::Matrix<_ScalarT, _n1, m_sc_nmatelems> B(np, y->dim0() * y->dimm() * y->dimm());
    
    std::size_t ix, i0, im0, im1;
    
    // Construct the coefficient matrix of the linear equations for the second derivative of the cubic spline S"(x)
    // First row, for the boundary condition y"(x_min) + y"(x_max) = G2
    A(0, 0) = 1.0;
    A(0, np - 1) = 1.0;
    for (ix = 1; ix < np - 1; ++ix) {
        A(ix, ix - 1) = ((*x)(ix) - (*x)(ix - 1)) / ((*x)(ix + 1) - (*x)(ix - 1));
        A(ix, ix + 1) = 1.0 - A(ix, ix - 1);
    }
    // Last row, for the boundary condition y'(x_min) + y'(x_max) = G1
    A(np - 1, 0) = -2.0;
    A(np - 1, 1) = -1.0;
    A(np - 1, np - 2) = ((*x)(np - 1) - (*x)(np - 2)) / ((*x)(1) - (*x)(0));
    A(np - 1, np - 1) = 2.0 * A(np - 1, np - 2);
    
    // Construct the rhs-value matrix
    for (i0 = 0; i0 < y->dim0(); ++i0) {
        for (im1 = 0; im1 < y->dimm(); ++im1) {
            for (im0 = 0; im0 < y->dimm(); ++im0) {
                B(0, flatIndex(i0, im0, im1)) = Ghfc(i0, 1, im0, im1);
                for (ix = 1; ix < np - 1; ++ix) {
                    B(ix, flatIndex(i0, im0, im1)) = 6.0 / ((*x)(ix + 1) - (*x)(ix - 1)) * (((*y)(i0, ix + 1, im0, im1) - (*y)(i0, ix, im0, im1)) / ((*x)(ix + 1) - (*x)(ix)) - ((*y)(i0, ix, im0, im1) - (*y)(i0, ix - 1, im0, im1)) / ((*x)(ix) - (*x)(ix - 1)));
                }
                B(np - 1, flatIndex(i0, im0, im1)) = 6.0 / ((*x)(1) - (*x)(0)) * (Ghfc(i0, 0, im0, im1) - ((*y)(i0, 1, im0, im1) - (*y)(i0, 0, im0, im1)) / ((*x)(1) - (*x)(0)) - ((*y)(i0, np - 1, im0, im1) - (*y)(i0, np - 2, im0, im1)) / ((*x)(np - 1) - (*x)(np - 2)));
            }
        }
    }
    
    // Solve for the second derivatives
    Eigen::PartialPivLU<Eigen::Matrix<double, _n1, _n1> > dec(A);
    if constexpr (std::is_same<_ScalarT, double>::value) m_deriv2 = dec.solve(B);
    else if (std::is_same<_ScalarT, std::complex<double> >::value) m_deriv2 = dec.solve(B.real()) + 1i * dec.solve(B.imag());
}

template <typename _ScalarT, int _n0, int _n1, int _nm>
_ScalarT CubicSplineMat<_ScalarT, _n0, _n1, _nm>::operator()(const std::size_t i0, const double x0, const std::size_t im0, const std::size_t im1) const {
    assert(x0 >= (*m_ptr2x)(0) && x0 <= (*m_ptr2x)(m_ptr2x->size() - 1));
    
    const std::size_t j0 = flatIndex(i0, im0, im1);
    std::size_t x0si = std::lower_bound(m_ptr2x->cbegin(), m_ptr2x->cend() - 1, x0) - m_ptr2x->cbegin();
    if (x0si > 0) --x0si;
    const double Dx0 = x0 - (*m_ptr2x)(x0si);
    const double Dx1 = (*m_ptr2x)(x0si + 1) - x0;
    const double dx = (*m_ptr2x)(x0si + 1) - (*m_ptr2x)(x0si);
    const double dxsq = dx * dx;
    
    return (m_deriv2(x0si, j0) * Dx1 * Dx1 * Dx1 / 6.0 + m_deriv2(x0si + 1, j0) * Dx0 * Dx0 * Dx0 / 6.0
            + ((*m_ptr2y)(i0, x0si, im0, im1) - m_deriv2(x0si, j0) * dxsq / 6.0) * Dx1
            + ((*m_ptr2y)(i0, x0si + 1, im0, im1) - m_deriv2(x0si + 1, j0) * dxsq / 6.0) * Dx0) / dx;
}

template <typename _ScalarT, int _n0, int _n1, int _nm>
_ScalarT CubicSplineMat<_ScalarT, _n0, _n1, _nm>::equidistAt(const std::size_t i0, const double x0, const std::size_t im0, const std::size_t im1) const {
    assert(x0 >= (*m_ptr2x)(0) && x0 <= (*m_ptr2x)(m_ptr2x->size() - 1));
    
    const std::size_t j0 = flatIndex(i0, im0, im1);
    const std::size_t x0si = std::min(static_cast<std::size_t>((x0 - (*m_ptr2x)(0)) / m_dx), static_cast<std::size_t>(m_ptr2x->size() - 2));
    const double Dx0 = x0 - (*m_ptr2x)(x0si);
    const double Dx1 = (*m_ptr2x)(x0si + 1) - x0;
    const double dxsq = m_dx * m_dx;
    
    return (m_deriv2(x0si, j0) * Dx1 * Dx1 * Dx1 / 6.0 + m_deriv2(x0si + 1, j0) * Dx0 * Dx0 * Dx0 / 6.0
            + ((*m_ptr2y)(i0, x0si, im0, im1) - m_deriv2(x0si, j0) * dxsq / 6.0) * Dx1
            + ((*m_ptr2y)(i0, x0si + 1, im0, im1) - m_deriv2(x0si + 1, j0) * dxsq / 6.0) * Dx0) / m_dx;
}

template <typename _ScalarT, int _n0, int _n1, int _nm>
std::complex<double> CubicSplineMat<_ScalarT, _n0, _n1, _nm>::fourierTransform(const std::size_t i0, const double omega, const std::size_t im0, const std::size_t im1) const {
    const std::size_t dim1 = m_ptr2x->size();
    const std::size_t j0 = flatIndex(i0, im0, im1);
    
    const std::complex<double> eiwtbegin = std::exp(1i * std::fmod(omega * (*m_ptr2x)(0), 2 * M_PI));
    const std::complex<double> eiwtend = std::exp(1i * std::fmod(omega * (*m_ptr2x)(dim1 - 1), 2 * M_PI));
    
    const _ScalarT deriv1begin = -((*m_ptr2x)(1) - (*m_ptr2x)(0)) / 6.0 * (2.0 * m_deriv2(0, j0) + m_deriv2(1, j0))
    + ((*m_ptr2y)(i0, 1, im0, im1) - (*m_ptr2y)(i0, 0, im0, im1)) / ((*m_ptr2x)(1) - (*m_ptr2x)(0));
    const _ScalarT deriv1end = ((*m_ptr2x)(dim1 - 1) - (*m_ptr2x)(dim1 - 2)) / 6.0 * (m_deriv2(dim1 - 2, j0) + 2.0 * m_deriv2(dim1 - 1, j0))
    + ((*m_ptr2y)(i0, dim1 - 1, im0, im1) - (*m_ptr2y)(i0, dim1 - 2, im0, im1)) / ((*m_ptr2x)(dim1 - 1) - (*m_ptr2x)(dim1 - 2));
    const _ScalarT deriv3begin = (m_deriv2(1, j0) - m_deriv2(0, j0)) / ((*m_ptr2x)(1) - (*m_ptr2x)(0));
    const _ScalarT deriv3end = (m_deriv2(dim1 - 1, j0) - m_deriv2(dim1 - 2, j0)) / ((*m_ptr2x)(dim1 - 1) - (*m_ptr2x)(dim1 - 2));

    // Approximate the residue of integration py parts, i.e., the integration of the 4th-order derivative, which is approximated by finite difference.
    // Not adding halves of the head and tail part because the fourth derivatives at the first and last grid point can not be approximated accurately.
    std::complex<double> integral(0.0, 0.0);
    for (std::size_t i = 1; i < dim1 - 1; ++i) {
        integral += std::exp(1i * std::fmod(omega * (*m_ptr2x)(i), 2 * M_PI)) * ((m_deriv2(i + 1, j0) - m_deriv2(i, j0)) / ((*m_ptr2x)(i + 1) - (*m_ptr2x)(i))
                                                                         - (m_deriv2(i, j0) - m_deriv2(i - 1, j0)) / ((*m_ptr2x)(i) - (*m_ptr2x)(i - 1)));
    }
    integral -= eiwtend * deriv3end - eiwtbegin * deriv3begin;   // Add boundary term of the 4th integration by parts
    integral /= omega * omega * omega * omega;
    // Add the boundary terms from 1st to 3rd integration by parts
    integral += (eiwtend * (*m_ptr2y)(i0, dim1 - 1, im0, im1) - eiwtbegin * (*m_ptr2y)(i0, 0, im0, im1)) / (1i * omega) - (eiwtend * deriv1end - eiwtbegin * deriv1begin) / (-omega * omega) + (eiwtend * m_deriv2(dim1 - 1, j0) - eiwtbegin * m_deriv2(0, j0)) / (-1i * omega * omega * omega);

    return integral;
}

// This method is automatically implicitly instantiated because it's inside a class.
// Eigen::Ref<Eigen::MatrixXcd> also references fixed size matrix.
template <typename _ScalarT, int _n0, int _n1, int _nm>
void CubicSplineMat<_ScalarT, _n0, _n1, _nm>::fourierTransform(const std::size_t i0, const double omega,
                                                               //Eigen::Ref<typename SqMatArray<_ScalarT, _n0, _n1, _nm>::DataType> result
                                                               Eigen::Ref<Eigen::Matrix<_ScalarT, Eigen::Dynamic, Eigen::Dynamic> > result) const {
    std::size_t im0, im1;
    result.resize(m_ptr2y->dimm(), m_ptr2y->dimm());  // For blocks, this just does runtime assertion that the new size matches the original size
    for (im1 = 0; im1 < m_ptr2y->dimm(); ++im1) {
        for (im0 = 0; im0 < m_ptr2y->dimm(); ++im0) result(im0, im1) = fourierTransform(i0, omega, im0, im1);
    }
}



typedef CubicSplineMat<double, 2, Eigen::Dynamic, Eigen::Dynamic> CubicSplineMat2XXd;
typedef CubicSplineMat<std::complex<double>, 2, Eigen::Dynamic, Eigen::Dynamic> CubicSplineMat2XXcd;

#endif /* cubic_spline_hpp */
