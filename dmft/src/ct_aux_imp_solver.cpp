//
//  ct_aux_imp_solver.cpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#include <iostream>
#include <iomanip>      // std::setw
#include <cmath>
#include <chrono>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include "ct_aux_imp_solver.hpp"

using namespace std::complex_literals;



NMatrix::NMatrix(std::shared_ptr<const BareGreenFunction> G0, const double gamma, const double K) : m_ptr2G0(G0), m_expg(std::exp(gamma)), m_K(K) {
    m_fermisign = 1.0;   // Zero vertex-order expansion is just 1, so its sign is 1.
}

// Try to insert the proposed vertex with a given barrier. Return the acceptance rate and whether accepted the insertion.
// tau is the position of its value in the discrete tau grid. aux_spin can be 1 or -1.
std::pair<double, bool> NMatrix::tryInsertVertex(const vertex& v, const double barrier) {
    assert(v.tau >=0 && v.tau < m_ptr2G0->inverseTemperature() && v.site < m_ptr2G0->nSites() && (v.aux_spin == 1 || v.aux_spin == -1));
    
    std::pair<double, bool> acceptance(0.0, false);
    std::array<std::complex<double>, 2> inv_diag, diag;  // Diagonal elements to be inserted to the (inverse) matrices for up and down spins
    int s;
     
    // Construct the diagonal element to be inserted to the inverse matrices
    for (s = 0; s < 2; ++s) inv_diag[s] = expV(s, v.aux_spin) - m_ptr2G0->valsOnTauGrid()(s, 0)(v.site, v.site) * (expV(s, v.aux_spin) - 1);  // Always a real number
    
    if (m_vertices.size() > 0) {
        std::array<Eigen::VectorXcd, 2> colt, inv_col;  // Columns to be inserted to the inverse matrices for up and down spins
        std::array<Eigen::RowVectorXcd, 2> inv_row;  // Rows to be inserted to the inverse matrices for up and down spins
        std::size_t i;
        
        // Construct the columns and rows to be inserted to the inverse matrices
        for (s = 0; s < 2; ++s) {
            inv_col[s].resize(m_vertices.size());
            inv_row[s].resize(m_vertices.size());
            for (i = 0; i < m_vertices.size(); ++i) {
                inv_col[s](i) = -m_ptr2G0->interpValAtExtendedTau(s, m_vertices[i].site, v.site, m_vertices[i].tau - v.tau) * (expV(s, v.aux_spin) - 1);
                inv_row[s](i) = -m_ptr2G0->interpValAtExtendedTau(s, v.site, m_vertices[i].site, v.tau - m_vertices[i].tau) * (expV(s, m_vertices[i].aux_spin) - 1);
            }
        }
        
        // Calculate acceptance rate
        for (s = 0; s < 2; ++s) {
            colt[s].resize(m_vertices.size());
            colt[s].noalias() = m_N[s] * inv_col[s];
            diag[s] = 1.0 / (inv_diag[s] - (inv_row[s] * colt[s])(0, 0));
        }
        acceptance.first = std::real((m_K / (m_vertices.size() + 1.0)) / (diag[0] * diag[1]));
        // Test
        // std::cout << "Check if real: " << diag[0] * diag[1] << std::endl;
        
        if ((barrier < 0 || barrier <= std::fabs(acceptance.first)) && barrier <= 1) {
            // Insertion is accepted
            Eigen::RowVectorXcd row(m_vertices.size());
            
            // Calculate the (diagonal elements) columns, rows, block matrices to construct the new matrices
            for (s = 0; s < 2; ++s) {
                // col = -colt1 * diag1;
                row.noalias() = -diag[s] * (inv_row[s] * m_N[s]);
                m_N[s].noalias() -= colt[s] * row;  // mat1 = mat1 - colt1 * row
                // Enlarge the matrices with the original matrices at the topleft corner untouched
                m_N[s].conservativeResize(m_vertices.size() + 1, m_vertices.size() + 1);
                // Perform insertion
                m_N[s].topRightCorner(m_vertices.size(), 1) = -colt[s] * diag[s];
                m_N[s].bottomLeftCorner(1, m_vertices.size()) = row;
                m_N[s](m_vertices.size(), m_vertices.size()) = diag[s];
            }
            
            // Add the new vertex to the configuration
            m_vertices.push_back(v);
            // Update Fermionic sign
            // fermi_sign *= (acceptance.first < 0) ? -1.0 : 1.0;
            m_fermisign *= copysign(1.0, acceptance.first);
            
            acceptance.second = true;
        }
    }
    else {
        // Current vertex order is zero
        for (s = 0; s < 2; ++s) diag[s] = 1.0 / inv_diag[s];
        acceptance.first = std::real(m_K / (diag[0] * diag[1]));
        
        if ((barrier < 0 || barrier <= std::fabs(acceptance.first)) && barrier <= 1) {
            // Insertion is accepted
            for (s = 0; s < 2; ++s) {
                m_N[s].resize(1, 1);
                m_N[s](0, 0) = diag[s];
            }
            
            // Add the new vertex to the configuration
            m_vertices.push_back(v);
            // Update Fermionic sign
            // fermi_sign *= (acceptance.first < 0) ? -1.0 : 1.0;
            m_fermisign *= copysign(1.0, acceptance.first);
            
            acceptance.second = true;
        }
    }
    
    return acceptance;
}

std::pair<double, bool> NMatrix::tryRemoveVertex(const std::size_t p, const double barrier) {
    std::pair<double, bool> acceptance(0.0, false);
    if (m_vertices.size() == 0) {
        return acceptance;
    }
    
    assert(p < m_vertices.size());
    
    int s;
    
    acceptance.first = std::real((m_vertices.size() / m_K) * m_N[0](p, p) * m_N[1](p, p));
    // Test
    // std::cout << "Check if real: " << N[0](p, p) * N[1](p, p) << std::endl;
    
    if ((barrier < 0 || barrier <= std::fabs(acceptance.first)) && barrier <= 1) {
        // Removal is accepted
        if (m_vertices.size() > 1) {
            if (p == 0) {
                for (s = 0; s < 2; ++s) {
                    // bmat = N1.bottomRightCorner(vertex_order - 1, vertex_order - 1);
                    // col = N1.bottomLeftCorner(vertex_order - 1, 1);
                    // row = N1.topRightCorner(1, vertex_order - 1);
                    // // N1.noalias() = bmat - (col * row) / diag1;
                    // N1 = std::move(bmat);  // Use move assignment because we know bmat is temporary
                    // N1.noalias() -= (col * row) / diag1;
                    m_N[s].bottomRightCorner(m_vertices.size() - 1, m_vertices.size() - 1).noalias() -= (m_N[s].bottomLeftCorner(m_vertices.size() - 1, 1) * m_N[s].topRightCorner(1, m_vertices.size() - 1)) / m_N[s](p, p);
                    m_N[s] = m_N[s].bottomRightCorner(m_vertices.size() - 1, m_vertices.size() - 1).eval();  // Call eval() to evaluate to temporary
                }
            }
            else if (p < m_vertices.size() - 1) {
                Eigen::MatrixXcd bmat;
                Eigen::VectorXcd col(m_vertices.size() - 1);
                Eigen::RowVectorXcd row(m_vertices.size() - 1);
                
                for (s = 0; s < 2; ++s) {
                    bmat.resize(m_vertices.size() - 1, m_vertices.size() - 1);   // Just before this resize bmat was empty because it was moved
                    bmat.topLeftCorner(p, p) = m_N[s].topLeftCorner(p, p);
                    bmat.topRightCorner(p, m_vertices.size() - p - 1) = m_N[s].topRightCorner(p, m_vertices.size() - p - 1);
                    bmat.bottomLeftCorner(m_vertices.size() - p - 1, p) = m_N[s].bottomLeftCorner(m_vertices.size() - p - 1, p);
                    bmat.bottomRightCorner(m_vertices.size() - p - 1, m_vertices.size() - p - 1) = m_N[s].bottomRightCorner(m_vertices.size() - p - 1, m_vertices.size() - p - 1);
                    col.head(p) = m_N[s].block(0, p, p, 1);
                    col.tail(m_vertices.size() - p - 1) = m_N[s].block(p + 1, p, m_vertices.size() - p - 1, 1);
                    row.head(p) = m_N[s].block(p, 0, 1, p);
                    row.tail(m_vertices.size() - p - 1) = m_N[s].block(p, p + 1, 1, m_vertices.size() - p - 1);
                    // N1.noalias() = bmat - (col * row) / diag1;
                    bmat.noalias() -= (col * row) / m_N[s](p, p);
                    m_N[s] = std::move(bmat);  // Use move assignment because we know bmat is temporary
                }
            }
            else if (p == m_vertices.size() - 1) {
                for (s = 0; s < 2; ++s) {
                    // // bmat = N1.topLeftCorner(vertex_order - 1, vertex_order - 1);
                    // col = N1.topRightCorner(vertex_order - 1, 1);
                    // row = N1.bottomLeftCorner(1, vertex_order - 1);
                    // //N1.noalias() = bmat - (col * row) / diag1;
                    // N1.conservativeResize(vertex_order - 1, vertex_order - 1);
                    // N1.noalias() -= (col * row) / diag1;
                    m_N[s].topLeftCorner(m_vertices.size() - 1, m_vertices.size() - 1).noalias() -= (m_N[s].topRightCorner(m_vertices.size() - 1, 1) * m_N[s].bottomLeftCorner(1, m_vertices.size() - 1)) / m_N[s](p, p);
                    m_N[s].conservativeResize(m_vertices.size() - 1, m_vertices.size() - 1);  // conservativeResize just removes the right-most column and bottom-most row
                }
            }
        }
        else if (m_vertices.size() == 1) {
            for (s = 0; s < 2; ++s) m_N[s].resize(0, 0);
        }
        
        // Remove the pth vertex
        m_vertices.erase(m_vertices.begin() + p);
        // Update Fermionic sign
        // fermi_sign *= (acceptance.first < 0) ? -1.0 : 1.0;
        m_fermisign *= copysign(1.0, acceptance.first);
        
        acceptance.second = true;
    }
    
    return acceptance;
}

std::pair<double, bool> NMatrix::tryShiftTau(const std::size_t p, const double tau, const double barrier) {
    std::pair<double, bool> acceptance(0.0, false);
    
    if (m_vertices.size() > 0) {
        assert(p < m_vertices.size() && tau >= 0 && tau <= m_ptr2G0->inverseTemperature());
        
        // Make a copy of the original N matrices and configuration
        std::array<Eigen::MatrixXcd, 2> N0 = m_N;
        std::vector<vertex> vertices0(m_vertices);
        double fermi_sign0 = m_fermisign;
        std::pair<double, bool> prob;
        double b = barrier;
        
        // Remove the vertex and record the acceptance rate
        prob = tryRemoveVertex(p, -1);
        acceptance.first = prob.first;
        
        if (barrier <= 1 && barrier >= 0) b = barrier / std::fabs(acceptance.first);
        
        // Try insert the new vertex and obtain the acceptance rate for the vertex shift.
        // Note that for 0 <= barrier <= 1, barrier < abs(shift rate) is equivalent to
        // barrier / abs(removal rate) < abs(insertion rate)
        vertex v = {tau, vertices0[p].site, vertices0[p].aux_spin};
        prob = tryInsertVertex(v, b);
        
        acceptance.first *= prob.first;  // This is formally the acceptance rate for the shift
        acceptance.second = prob.second; // This is formally whether the shift is accepted
        
        if (!acceptance.second) {
            // Shift is not accepted; restore the matrices and configuration
            m_N = std::move(N0);  // Use move assignment because N0 is temporary
            m_vertices = std::move(vertices0);
            m_fermisign = fermi_sign0;
        }
    }
    
    return acceptance;
}

std::pair<double, bool> NMatrix::tryShiftSite(const std::size_t p, const std::size_t site, const double barrier) {
    std::pair<double, bool> acceptance(0.0, false);
    
    if (m_vertices.size() > 0) {
        assert(p < m_vertices.size() && site < m_ptr2G0->nSites());
        
        // Make a copy of the original matrices and configuration
        std::array<Eigen::MatrixXcd, 2> N0 = m_N;
        std::vector<vertex> vertices0(m_vertices);
        double fermi_sign0 = m_fermisign;
        std::pair<double, bool> prob;
        double b = barrier;
        
        // Remove the vertex and record the acceptance rate
        prob = tryRemoveVertex(p, -1);
        acceptance.first = prob.first;
        
        if (barrier <= 1 && barrier >= 0) b = barrier / std::fabs(acceptance.first);
        
        // Try insert the new vertex and obtain the acceptance rate for the vertex shift.
        // Note that for 0 <= barrier <= 1, barrier < abs(shift rate) is equivalent to
        // barrier / abs(removal rate) < abs(insertion rate)
        vertex v = {vertices0[p].tau, site, vertices0[p].aux_spin};
        prob = tryInsertVertex(v, b);
        
        acceptance.first *= prob.first;  // This is formally the acceptance rate for the shift
        acceptance.second = prob.second; // This is formally whether the shift is accepted
        
        if (!acceptance.second) {
            // Shift is not accepted; restore the matrices and configuration
            m_N = std::move(N0);   // Use move assignment because N0 is temporary
            m_vertices = std::move(vertices0);
            m_fermisign = fermi_sign0;
        }
    }
    
    return acceptance;
}

// Try flipping auxiliary spin; this has special simple formulae
std::pair<double, bool> NMatrix::tryFlipAuxSpin(const std::size_t p, const double barrier) {
    std::pair<double, bool> acceptance(0.0, false);
    
    if (m_vertices.size() > 0) {
        assert(p < m_vertices.size());
        
        std::array<double, 2> eV, gam;
        std::array<std::complex<double>, 2> NG0, R;
        int s;
        
        for (s = 0; s < 2; ++s) {
            eV[s] = expV(s, m_vertices[p].aux_spin);
            gam[s] = 1.0 / (eV[s] * eV[s]) - 1.0;
            NG0[s] = (m_N[s](p, p) * eV[s] - 1.0) / (eV[s] - 1.0);
            R[s] = 1.0 + (1.0 - NG0[s]) * gam[s];
        }
        acceptance.first = std::real(R[0] * R[1]);
        // Test
        // std::cout << "Check if real: " << R[0] * R[1] << std::endl;
        
        if ((barrier < 0 || barrier <= std::fabs(acceptance.first)) && barrier <= 1) {
            // Flip is accepted
            Eigen::VectorXcd col(m_vertices.size());
            Eigen::RowVectorXcd row;
            std::complex<double> lam;
//            std::size_t i;
            
            for (s = 0; s < 2; ++s) {
                lam = gam[s] / R[s];
//                for (i = 0; i < vertices.size(); ++i) {
//                    if (i == p) col(i) = (NG0[s] - 1.0) * lam;
//                    else col(i) = N[s](i, p) * eV[s] / (eV[s] - 1.0) * lam;
//                }
                col = (eV[s] / (eV[s] - 1.0) * lam) * m_N[s].col(p);
                col(p) = (NG0[s] - 1.0) * lam;
                row = m_N[s].row(p);  // If not made this tmp, the implicit tmp in the next step will be of the size of N[s]
                m_N[s].noalias() += col * row;  // mat1 = mat1 + col * row
            }
            
            // Flip the auxiliary spin
            m_vertices[p].aux_spin *= -1;
            // Update Fermionic sign
            // fermi_sign *= (acceptance.first < 0) ? -1.0 : 1.0;
            m_fermisign *= copysign(1.0, acceptance.first);
            
            acceptance.second = true;
        }
    }
    
    return acceptance;
}

void NMatrix::reset() {
    m_vertices.clear();
    for (int s = 0; s < 2; ++s) m_N[s].resize(0, 0);
    m_fermisign = 1.0;
}

void NMatrix::setParams(const double gamma, const double K) {
    m_expg = std::exp(gamma);
    m_K = K;
}






ImpurityProblem::ImpurityProblem(std::shared_ptr<const BareHamiltonian> H0_, std::shared_ptr<const BareGreenFunction> G0_, const double U_, const double K_, std::shared_ptr<GreenFunction> G_) : H0(H0_), G0(G0_), U(U_), K(K_), G(G_) { }






// Measure and accumulate G(iw) corrections to G0(iw), Gc(iw), i.e., G(iw) = G0(iw) + Gc(iw). We directly accumulate Gc(iw) to variable Gw.
void CTAUXImpuritySolver::measAccumGFfCoeffsCorr() {
    assert(m_ptr2problem->G->fourierCoeffs().dim0() == m_ptr2problem->G0->fourierCoeffs().dim0());
    if (m_ptr2problem->G0->tauGridSizeOfExpiwt() < 2) throw std::invalid_argument("Tau grid size of expiwt array is less than 2 so cannot utilize pre-calculated exp(iwt) to measure Gc(iw)!");
    
    const std::size_t nfcut = m_ptr2problem->G->freqCutoff();
    if (nfcut != m_ptr2problem->G0->freqCutoff()) throw std::range_error( "Frequency cutoffs of G and G0 do not match!" );
    const std::size_t nc = m_ptr2problem->G->nSites();
    if (nc != m_ptr2problem->G0->nSites()) throw std::range_error( "Numbers of sites of G and G0 do not match!" );
    const double beta = m_ptr2problem->G0->inverseTemperature();
    if (std::fabs(beta - m_ptr2problem->G->inverseTemperature()) > 1e-9) throw std::range_error( "Temperatures of G and G0 do not match!" );
    
    Eigen::VectorXd eV_1(m_vertices.size());
    Eigen::MatrixXcd M(m_vertices.size(), m_vertices.size()), G0l(nc, m_vertices.size()), G0r(m_vertices.size(), nc), MG(m_vertices.size(), nc), gc(nc, nc);
    int s;
    std::size_t o, p;
    
    Eigen::Array<std::size_t, Eigen::Dynamic, 1> tau_ind4eiwt(m_vertices.size());
    const double dtau_eiwt = beta / (m_ptr2problem->G0->tauGridSizeOfExpiwt() - 1);
    
    const int nrandmeasure = 10;
    Eigen::ArrayXXd randtaudiffs(m_vertices.size(), nrandmeasure), beta_randtaudiffs(m_vertices.size(), nrandmeasure), sgns(m_vertices.size(), nrandmeasure);
    int rm;
    std::size_t x;
    Eigen::VectorXd denssample(nc);
    double rdt;
    
    // Mount tau to the nearest grid point of expiwt array to use pre-calculated exp(iwt)
    for (p = 0; p < m_vertices.size(); ++p) tau_ind4eiwt(p) = static_cast<std::size_t>(m_vertices[p].tau / dtau_eiwt + 0.5);
    
    // Set random tau differences; same for up and down spins
    for (rm = 0; rm < nrandmeasure; ++rm) {
        rdt = beta * m_urd(m_reng);
        for (p = 0; p < m_vertices.size(); ++p) {
            randtaudiffs(p, rm)  = m_vertices[p].tau - rdt;
            if (randtaudiffs(p, rm) < 0) {
                sgns(p, rm) = -1.0;
                beta_randtaudiffs(p, rm) = -randtaudiffs(p, rm);  // beta_randtaudiffs equals beta subtracting rounded randtaudiffs
            }
            else {
                sgns(p, rm) = 1.0;
                beta_randtaudiffs(p, rm) = beta - randtaudiffs(p, rm);   // beta_randtaudiffs equals beta subtracting rounded randtaudiffs
            }
        }
    }
    
    for (s = 0; s < 2; ++s) {
        denssample.setZero();
        
        // Construct M matrix
//        for (l = 0; l < vertices.size(); ++l) {
//            M.row(l) = (expV(s, vertices[l].aux_spin) - 1.0) * N[s].row(l);
//        }
        for (p = 0; p < m_vertices.size(); ++p) eV_1(p) = expV(s, m_vertices[p].aux_spin) - 1.0;
        M.noalias() = eV_1.asDiagonal() * m_N[s];
        
        for (o = 0; o <= nfcut; ++o) {
            // Construct the Green's function matrices
            for (p = 0; p < m_vertices.size(); ++p)
                G0l.col(p) = m_ptr2problem->G0->fourierCoeffs()(s, o).col(m_vertices[p].site) * m_ptr2problem->G0->expiwt(tau_ind4eiwt(p), o);
            for (x = 0; x < nc; ++x) {
                for (p = 0; p < m_vertices.size(); ++p)
                    G0r(p, x) = m_ptr2problem->G0->fourierCoeffs()(s, o, m_vertices[p].site, x) / m_ptr2problem->G0->expiwt(tau_ind4eiwt(p), o);
            }
            MG.noalias() = M * G0r;
            gc.noalias() = (G0l * MG) / beta;
            // _problem->G->fourierCoeffs()(s, o) += fermi_sign * gc;
            m_ptr2problem->G->fourierCoeffs()(s, o).noalias() += m_fermisign * gc;
            m_ptr2problem->G->fCoeffsVar()(s, o) += m_fermisign * gc.cwiseAbs2();  // Accumulate squared norm of sample for calculating variance later
        }
        
        // Measure electron density, using time translational invariance
        for (rm = 0; rm < nrandmeasure; ++rm) {
            for (x = 0; x < nc; ++x) {
                for (p = 0; p < m_vertices.size(); ++p) G0r(p, x) = m_ptr2problem->G0->interpValAtExtendedTau(s, m_vertices[p].site, x, randtaudiffs(p, rm));
            }
            MG.noalias() = M * G0r;
            for (x = 0; x < nc; ++x) {
                for (p = 0; p < m_vertices.size(); ++p) denssample(x) += std::real(sgns(p, rm) * m_ptr2problem->G0->interpValAtExtendedTau(s, x, m_vertices[p].site, beta_randtaudiffs(p, rm)) * MG(p, x));
            }
        }
        denssample /= nrandmeasure;
        m_ptr2problem->G->elecDensities().col(s) += m_fermisign * denssample;
        m_ptr2problem->G->elecDensStdDev().col(s) += m_fermisign * denssample.cwiseAbs2();
    }
    
    m_measuredfermisign += m_fermisign;
    
    ++m_nmeasure;
}

// Measure and accumulate S(tau)
void CTAUXImpuritySolver::measAccumSelfEgf() {
    assert(m_ptr2problem->G->fourierCoeffs().dim0() == m_ptr2problem->G0->fourierCoeffs().dim0());
    
//    if (_problem->G0->tauGridSizeOfExpiwt() < _problem->G->nTauBins4selfEgf() + 1) {
//        throw std::range_error("The number of tau bins of the pre-calculated exp(i*w*t) array is smaller than that of S so cannot use the exp(i*w*t) array to accurately measure S!");
//    }
    const std::size_t nc = m_ptr2problem->G->nSites();
    if (nc != m_ptr2problem->G0->nSites()) throw std::range_error("Numbers of sites of G and G0 do not match!");
    const double beta = m_ptr2problem->G0->inverseTemperature();
    if (std::fabs(beta - m_ptr2problem->G->inverseTemperature()) > 1e-9) throw std::range_error("Temperatures of G and G0 do not match!");
    
    const std::size_t nbins4S = m_ptr2problem->G->nTauBins4selfEnGF();
    const double binsize4S = beta / nbins4S;
    // const double dtau4eiwt = beta / (_problem->G0->tauGridSizeOfExpiwt() - 1);
    Eigen::VectorXd eV_1(m_vertices.size());
    Eigen::MatrixXcd M(m_vertices.size(), m_vertices.size()), G0r(m_vertices.size(), nc), MG(m_vertices.size(), nc);
    const int nrandmeasure = 10;
    Eigen::ArrayXXd randtaudiffs(m_vertices.size(), nrandmeasure), beta_randtaudiffs(m_vertices.size(), nrandmeasure), sgns(m_vertices.size(), nrandmeasure);
    Eigen::Array<std::size_t, Eigen::Dynamic, Eigen::Dynamic> ibins4S(m_vertices.size(), nrandmeasure);
    int rm;
    double rdt;
    std::size_t p, x;
    Eigen::VectorXd denssample(nc);
    
    // Set random tau differences; same for up and down spins
    for (rm = 0; rm < nrandmeasure; ++rm) {
        rdt = beta * m_urd(m_reng);
        for (p = 0; p < m_vertices.size(); ++p) {
            randtaudiffs(p, rm)  = m_vertices[p].tau - rdt;
            if (randtaudiffs(p, rm) < 0) {
                sgns(p, rm) = -1.0;
                // ibin4S = std::min(static_cast<std::size_t>(taudiff / binsize4S), nbins4S - 1);  // taudiff here is always non-negative
                // Note the rounded randtaudiffs is always less than beta, so this cast is always less than or equal to nbins4S - 1
                ibins4S(p, rm) = static_cast<std::size_t>((randtaudiffs(p, rm) + beta) / binsize4S);
                beta_randtaudiffs(p, rm) = -randtaudiffs(p, rm);  // beta_randtaudiffs equals beta subtracting rounded randtaudiffs
            }
            else {
                sgns(p, rm) = 1.0;
                ibins4S(p, rm) = static_cast<std::size_t>(randtaudiffs(p, rm) / binsize4S);
                beta_randtaudiffs(p, rm) = beta - randtaudiffs(p, rm);   // beta_randtaudiffs equals beta subtracting rounded randtaudiffs
            }
        }
    }
    
    for (int s = 0; s < 2; ++s) {
        denssample.setZero();
        
//        for (p = 0; p < vertices.size(); ++p) {
//            M.row(p) = (expV(s, vertices[p].aux_spin) - 1) * N[s].row(p);
//        }
        for (p = 0; p < m_vertices.size(); ++p) eV_1(p) = expV(s, m_vertices[p].aux_spin) - 1.0;
        M.noalias() = eV_1.asDiagonal() * m_N[s];
        
        for (rm = 0; rm < nrandmeasure; ++rm) {
            for (x = 0; x < nc; ++x) {
                for (p = 0; p < m_vertices.size(); ++p) G0r(p, x) = m_ptr2problem->G0->interpValAtExtendedTau(s, m_vertices[p].site, x, randtaudiffs(p, rm));
            }
            MG.noalias() = M * G0r;
            
            // Accumulate S(tau_z), and because of the delta function in S, tau_z = tau_p - randtau.
            // Also there is a sign arising from the left-hand G0(tau + randtau - tau_p) = G0(tau - tau_z), and
            // we can absorb the sign into S.
            for (x = 0; x < nc; ++x) {
                for (p = 0; p < m_vertices.size(); ++p) {
                    m_ptr2problem->G->selfEnGF()(s, ibins4S(p, rm))(m_vertices[p].site, x) += (m_fermisign * sgns(p, rm) / (binsize4S * nrandmeasure)) * MG(p, x);
                    denssample(x) += std::real(sgns(p, rm) * m_ptr2problem->G0->interpValAtExtendedTau(s, x, m_vertices[p].site, beta_randtaudiffs(p, rm)) * MG(p, x));
                }
            }
        }
        denssample /= nrandmeasure;
        m_ptr2problem->G->elecDensities().col(s) += m_fermisign * denssample;
        m_ptr2problem->G->elecDensStdDev().col(s) += m_fermisign * denssample.cwiseAbs2();
    }
    
    m_measuredfermisign += m_fermisign;
    
    ++m_nmeasure;
}

std::pair<int, std::pair<double, bool> > CTAUXImpuritySolver::move1markovStepEqualProp() {
    const std::size_t nc = m_ptr2problem->G0->nSites();
    const double beta = m_ptr2problem->G0->inverseTemperature();
    std::pair<int, std::pair<double, bool> > info;
    std::pair<double, bool> acceptance;
    double barrier = m_urd(m_reng);
    
    if (m_vertices.size() == 0) {
        // Try to insert a vertex
        const vertex v = {m_urd(m_reng) * beta, static_cast<std::size_t>(m_urd(m_reng) * nc), static_cast<int>(m_urd(m_reng) * 2) * 2 - 1}; // auxiliary spin is -1 or 1
        
        acceptance = tryInsertVertex(v, barrier);
        info.first = 0;
    }
    else {
        double propose = m_urd(m_reng);
        
        if (propose < 1.0 / 3.0) {
            // Try to insert a vertex
            const vertex v = {m_urd(m_reng) * beta, static_cast<std::size_t>(m_urd(m_reng) * nc), static_cast<int>(m_urd(m_reng) * 2) * 2 - 1};
            
            acceptance = tryInsertVertex(v, barrier);
            info.first = 0;
        }
        else if (propose < 2.0 / 3.0) {
            // Try to remove a vertex
            const auto p = static_cast<std::size_t>(m_urd(m_reng) * m_vertices.size());
            
            acceptance = tryRemoveVertex(p, barrier);
            info.first = 1;
        }
        else {
            // Try shift a vertex
            const auto shifttausite = std::any_cast<bool>(parameters["shift t&site"]);
            const auto p = static_cast<std::size_t>(m_urd(m_reng) * m_vertices.size());
            double cp = 1.0 / 3.0;
            if (nc == 1) {
                cp = 0.5;
            }
            double w2s = m_urd(m_reng);
            
            if (shifttausite && w2s < cp) {
                // Try shift tau
                acceptance = tryShiftTau(p, m_urd(m_reng) * beta, barrier);
                info.first = 2;
            }
            else if (shifttausite && nc > 1 && w2s < 2.0 / 3.0) {
                // Try shift site
                auto site = static_cast<std::size_t>(m_urd(m_reng) * (nc - 1));
                if (site >= m_vertices[p].site) {
                    ++site;
                }
                acceptance = tryShiftSite(p, site, barrier);
                info.first = 3;
            }
            else {
                // Flip auxiliary spin
                acceptance = tryFlipAuxSpin(p, barrier);
                info.first = 4;
            }
        }
    }
    
    info.second = acceptance;
    
    ++m_nmarkovstep;
    
    return info;
}

void CTAUXImpuritySolver::reset() {
    NMatrix::reset();
    m_measuredfermisign = 0.0;
    m_nmarkovstep = 0;
    m_nmeasure = 0;
    m_avevertorder = 0;
    m_histogram.setZero();
}

CTAUXImpuritySolver::CTAUXImpuritySolver(std::shared_ptr<ImpurityProblem> problem) : m_ptr2problem(problem), NMatrix(problem->G0, acosh(1 + problem->U * problem->G0->inverseTemperature() * problem->G0->nSites() / (2 * problem->K)), problem->K), m_urd(0.0, 1.0) {
    // reset counters and vertex expansion although not necessary because they are reset at the beginning of solve() function
    reset();
    
    // Reserve space for the configuration vectors to help prevent run-time reallocation
    m_vertices.reserve(500);
    
    // Default the solver parameters; std::map just adds the element if it does not exist
    std::random_device rd;
    m_oldseed = rd() + problem->G->fourierCoeffs().processRank() * 137u;
    parameters["markov chain length"] = static_cast<std::size_t>(20000000);
    parameters["QMC time limit"] = 8.0;   // Unit is minute
    parameters["#warm up steps"] = static_cast<std::size_t>(10000);
    parameters["measure period"] = static_cast<std::size_t>(200);
    parameters["verbosity"] = std::string("progress_bar");   // Use std::string constructor to make the char array be of type std::string
    parameters["random seed"] = m_oldseed;
    parameters["histogram max order"] = static_cast<std::size_t>(100);
    parameters["shift t&site"] = false;
    parameters["does measure"] = true;
    parameters["measure what"] = std::string("S");
    parameters["magnetic order"] = std::string("paramagnetic");  // Or ferromagnetic or antiferromagnetic
    m_reng.seed(m_oldseed);
}

double CTAUXImpuritySolver::solve() {
    std::pair<int, std::pair<double, bool> > mstepinfo;
    int s;
    double simpinterror = -1.0;
    
    const auto neffstep_globtot = std::any_cast<std::size_t>(parameters.at("markov chain length"));
    const auto warmup = std::any_cast<std::size_t>(parameters.at("#warm up steps"));
    const auto measureperiod = std::any_cast<std::size_t>(parameters.at("measure period"));
    
    const std::size_t lsize = neffstep_globtot / m_ptr2problem->G->fourierCoeffs().processSize();
    const std::size_t neffstep = (m_ptr2problem->G->fourierCoeffs().processRank() < m_ptr2problem->G->fourierCoeffs().processSize() - 1) ? lsize : lsize + neffstep_globtot % m_ptr2problem->G->fourierCoeffs().processSize();
    
    const auto verbosity = std::any_cast<std::string>(parameters.at("verbosity"));
    // Hide cursor
    if (verbosity == "progress_bar" && m_ptr2problem->G->fourierCoeffs().processRank() == 0) indicators::show_console_cursor(false);
    
    // Create a progress bar
    indicators::BlockProgressBar bar{
        indicators::option::BarWidth{30},
        indicators::option::ForegroundColor{indicators::Color::green},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::PrefixText{"    Markov walking on process " + std::to_string(m_ptr2problem->G->fourierCoeffs().processRank()) + " "},
        indicators::option::MaxProgress{warmup + neffstep},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}
    };
    
    // Set random number seed and allocate histogram
    const auto newseed = std::any_cast<unsigned int>(parameters.at("random seed"));
    if (newseed != m_oldseed) {
        m_reng.seed(newseed);
        m_oldseed = newseed;
    }
    m_histogram.resize(std::any_cast<std::size_t>(parameters.at("histogram max order")));  // No-op if sizes match
    
    // Reset counters and vertex expansion
    reset();
    
    // Reset accumulators
    const auto does_measure = std::any_cast<bool>(parameters.at("does measure"));
    const auto measure_what = std::any_cast<std::string>(parameters.at("measure what"));
    if (does_measure) {
        m_ptr2problem->G->elecDensities().setZero();
        m_ptr2problem->G->elecDensStdDev().setZero();
        m_ptr2problem->G->fCoeffsVar()().setZero();
        if (measure_what == "S") m_ptr2problem->G->selfEnGF()().setZero();
        else if (measure_what == "G") m_ptr2problem->G->fourierCoeffs()().setZero();
        else throw std::invalid_argument("There are only two options for what to measure (S or G)!");
    }
    
    if (verbosity == "on" && m_ptr2problem->G->fourierCoeffs().processRank() == 0) {
        std::cout << std::setw(20) << "--------------------" << std::setw(35) << " Markov movement from root process " << std::setw(20)
        << "--------------------" << std::endl;
        std::cout << std::setw(15) << "#Markov step" << std::setw(15) << "Move type" << std::setw(15) << "Accept rate" << std::setw(15) << "Is updated"
        << std::setw(15) << "Vertex order" << std::endl;
        std::cout << std::setw(15) << "------------" << std::setw(15) << "---------" << std::setw(15) << "-----------" << std::setw(15) << "----------"
        << std::setw(15) << "------------" << std::endl;
    }
    
    const auto qmcdurlim = std::any_cast<double>(parameters.at("QMC time limit"));
    auto qmcbegin = std::chrono::high_resolution_clock::now();
    auto qmcend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<60> > qmcduration = qmcend - qmcbegin;  // Unit is minutes
    
    // QMC Markov random walk starts here
    for (std::size_t step = 0; step < warmup + neffstep && qmcduration.count() <= qmcdurlim; ++step) {
        // mstepinfo = move1markovStep();
        mstepinfo = move1markovStepEqualProp();
        
        if (verbosity == "on" && m_ptr2problem->G->fourierCoeffs().processRank() == 0) {
            std::cout << std::setw(15) << m_nmarkovstep << std::setw(15) << mstepinfo.first << std::setw(15) << mstepinfo.second.first << std::setw(15)
            << mstepinfo.second.second << std::setw(15) << m_vertices.size() << std::endl;
        }
        
        // Measure
        if (does_measure && step >= warmup && (step - warmup) % measureperiod == 0) {
            // Count average vertex order and vertex order histogram
            m_avevertorder += m_vertices.size();
            if (m_vertices.size() < m_histogram.size()) ++m_histogram[m_vertices.size()];
            
            if (measure_what == "S") measAccumSelfEgf();
            else if (measure_what == "G") measAccumGFfCoeffsCorr();
        }
        
        // Calculate duration of the QMC Markov random walk
        qmcend = std::chrono::high_resolution_clock::now();
        qmcduration = qmcend - qmcbegin;
        
        // Update progress bar
        if (verbosity == "progress_bar" && m_ptr2problem->G->fourierCoeffs().processRank() == 0) bar.tick();
    }
    // QMC Markov random walk ends here
    
    if (verbosity == "progress_bar" && m_ptr2problem->G->fourierCoeffs().processRank() == 0) {
        bar.set_option(indicators::option::PrefixText{"    Markov walking done ✔ "});
        // indicators::erase_line();  // Erase the completed bar. Don't do mark_as_completed because that will start a new line.
        bar.mark_as_completed();
        indicators::show_console_cursor(true);  // Show cursor
    }
    
    if (verbosity == "on" && m_ptr2problem->G->fourierCoeffs().processRank() == 0)
        std::cout << std::setw(22) << "----------------------" << std::setw(16) << " #measurements: " << std::setw(15) << m_nmeasure << std::setw(22)
        << "----------------------" << std::endl;
    
    // Combine counters on all processes
    MPI_Allreduce(MPI_IN_PLACE, &m_measuredfermisign, 1, MPI_DOUBLE, MPI_SUM, m_ptr2problem->G->fourierCoeffs().mpiCommunicator());
    MPI_Allreduce(MPI_IN_PLACE, &m_nmeasure, 1, my_MPI_SIZE_T, MPI_SUM, m_ptr2problem->G->fourierCoeffs().mpiCommunicator());
    MPI_Allreduce(MPI_IN_PLACE, &m_nmarkovstep, 1, my_MPI_SIZE_T, MPI_SUM, m_ptr2problem->G->fourierCoeffs().mpiCommunicator());
    MPI_Allreduce(MPI_IN_PLACE, &m_avevertorder, 1, MPI_DOUBLE, MPI_SUM, m_ptr2problem->G->fourierCoeffs().mpiCommunicator());
    MPI_Allreduce(MPI_IN_PLACE, m_histogram.data(), static_cast<int>(m_histogram.size()), my_MPI_SIZE_T, MPI_SUM, m_ptr2problem->G->fourierCoeffs().mpiCommunicator());
    m_measuredfermisign /= m_nmeasure;
    m_avevertorder /= m_nmeasure;
    
    if (verbosity == "on" && m_ptr2problem->G->fourierCoeffs().processRank() == 0)
        std::cout << std::setw(15) << "---------------" << std::setw(29) << " #global total measurements: " << std::setw(15) << m_nmeasure
        << std::setw(16) << "----------------" << std::endl;
    
    // Finalize measurement
    if (does_measure) {
        // Finalize the measurement of electron density
        MPI_Allreduce(MPI_IN_PLACE, m_ptr2problem->G->elecDensities().data(), static_cast<int>(m_ptr2problem->G->elecDensities().size()), MPI_DOUBLE, MPI_SUM, m_ptr2problem->G->fourierCoeffs().mpiCommunicator());
        MPI_Allreduce(MPI_IN_PLACE, m_ptr2problem->G->elecDensStdDev().data(), static_cast<int>(m_ptr2problem->G->elecDensStdDev().size()), MPI_DOUBLE, MPI_SUM, m_ptr2problem->G->fourierCoeffs().mpiCommunicator());
        m_ptr2problem->G->elecDensities() /= m_nmeasure * m_measuredfermisign;
        m_ptr2problem->G->elecDensStdDev() = ((m_ptr2problem->G->elecDensStdDev() / (m_nmeasure * m_measuredfermisign) - m_ptr2problem->G->elecDensities().cwiseAbs2()) / (m_nmeasure - 1)).cwiseSqrt();
        for (s = 0; s < 2; ++s) m_ptr2problem->G->elecDensities().col(s) += m_ptr2problem->G0->valsOnTauGrid()(s, m_ptr2problem->G0->tauGridSize() - 1).diagonal().real();
        
        // Use measured electron densities to compute G's high-frequency expansion coefficients
        m_ptr2problem->G->computeMoments(*(m_ptr2problem->H0), m_ptr2problem->U);
        
        if (measure_what == "S") {
            // Combine accumulated measurements of S on all processes and process them
            m_ptr2problem->G->selfEnGF().allSum();   // All processes need the full-size S
            m_ptr2problem->G->selfEnGF()() /= m_nmeasure * m_measuredfermisign;  // Average value of measured quantity
            
            // Each process only evaluate G(iw) on its mastered imaginary partition; return an estimation of Simpson integration error
            // for obtaining electron densities
            simpinterror = m_ptr2problem->G->evalFromSelfEnGF(*(m_ptr2problem->G0));
        }
        else if (measure_what == "G") {
            auto Gwmastpart = m_ptr2problem->G->fourierCoeffs().mastFlatPart();
            auto Gwvarmastpart = m_ptr2problem->G->fCoeffsVar().mastFlatPart();
            // Combine measurements on every processes. Each process only holds valid data on its mastered imaginary partition.
            Gwmastpart.sum2mastPart();
            Gwvarmastpart.sum2mastPart();
            
            Gwmastpart() /= m_nmeasure * m_measuredfermisign;  // This is G's correction to G0
            Gwvarmastpart() = (Gwvarmastpart() / (m_nmeasure * m_measuredfermisign) - Gwmastpart().cwiseAbs2()) / (m_nmeasure - 1);
            Gwmastpart() += m_ptr2problem->G0->fourierCoeffs().mastFlatPart()();  // Add G0 to obtain G
            
            m_ptr2problem->G->invFourierTrans();  // Not required for simulation, maybe required in future versions, but just for output for now
        }
        const auto magneticorder = std::any_cast<std::string>(parameters.at("magnetic order"));
        if (magneticorder == "paramagnetic") m_ptr2problem->G->symmetrizeSpins(true);  // All data gathered first
    }
    return simpinterror;  // -1 means not used Simpson integration and thus not estimate the error
}

void CTAUXImpuritySolver::updateInterPhysParams() {
    double gamma = acosh(1 + m_ptr2problem->U * m_ptr2problem->G0->inverseTemperature() * m_ptr2problem->G0->nSites() / (2 * m_ptr2problem->K));
    // Update inherited parameters
    setParams(gamma, m_ptr2problem->K);
}

