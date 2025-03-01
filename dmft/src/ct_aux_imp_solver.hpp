//
//  ct_aux_imp_solver.hpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#ifndef ct_aux_imp_solver_hpp
#define ct_aux_imp_solver_hpp

#include <memory>
#include <random>
#include <map>
#include <any>
#include "green_function.hpp"



struct vertex {
    double tau;
    Eigen::Index site;   // Index of a site
    int aux_spin;
};


class NMatrix {
private:
    std::shared_ptr<const BareGreenFunction> m_ptr2G0;   // Make a copy of pointer for self-usage
    double m_expg;
    double m_K;
    
    struct cut {
        Eigen::Index size() const {return shrunk_size;}
        Eigen::Index operator[](Eigen::Index i) const {return i < cut_ind ? i : i + 1;}
        Eigen::Index shrunk_size, cut_ind;
    };
    
protected:
    std::vector<vertex> m_vertices;
    // This is good because the size of Eigen::MatrixXcd is known at compile time and its dynamic memory is allocated on heap
    std::array<Eigen::MatrixXcd, 2> m_N;  // Current matrices for up and down spins
    double m_fermisign;   // Fermionic sign
    
    // Cheap calculation of e^V matrix elements. spin = 0, 1; aux_spin = -1, 1. spin must be of int type because it is involved in substraction.
    double expV(const int spin, const int aux_spin) const {return ((1 - 2 * spin) * aux_spin > 0) ? m_expg : 1.0 / m_expg;}
    
public:
    NMatrix() : m_expg(0.0), m_K(0.0), m_fermisign(1.0) {}
    NMatrix(std::shared_ptr<const BareGreenFunction> G0, const double gamma, const double K);
    NMatrix(const NMatrix&) = default;
    NMatrix(NMatrix&&) = default;
    NMatrix& operator=(const NMatrix&) = default;
    NMatrix& operator=(NMatrix&&) = default;
    
    virtual ~NMatrix() {}
    
    std::pair<double, bool> tryInsertVertex(const vertex& v, const double barrier);
    
    std::pair<double, bool> tryRemoveVertex(const Eigen::Index p, const double barrier);
    
    std::pair<double, bool> tryShiftTau(const Eigen::Index p, const double tau, const double barrier);
    
    std::pair<double, bool> tryShiftSite(const Eigen::Index p, const Eigen::Index site, const double barrier);
    
    std::pair<double, bool> tryFlipAuxSpin(const Eigen::Index p, const double barrier);
    
    virtual void reset();
    
    void setParams(const double gamma, const double K);
};


class CTAUXImpuritySolver : protected NMatrix {
private:
    std::shared_ptr<const BareHamiltonian> m_ptr2H0;
    std::shared_ptr<const BareGreenFunction> m_ptr2G0;
    double m_U;
    double m_K;
    std::shared_ptr<GreenFunction> m_ptr2G;
    unsigned int m_oldseed;  // Keep a copy of the old random number seed, to help determine whether to re-seed the random number engine
    
protected:
    double m_measuredfermisign;   // Measured Fermionic sign
    // Use size_t because these QMC-related numbers could be large
    Eigen::Index m_nmarkovstep, m_nmeasure;
    double m_avevertorder;
    ArrayXindex m_histogram;
    std::mt19937 m_reng;   // Random number engine
    std::uniform_real_distribution<double> m_urd;  // Define random number distribution
    
    void measAccumGFfCoeffsCorr();
    
    void measAccumSelfEgf();
    
    std::pair<int, std::pair<double, bool> > move1markovStepEqualProp();
    
public:
    std::map<std::string, std::any> parameters;
    
    void reset();
    
    CTAUXImpuritySolver(std::shared_ptr<const BareHamiltonian> H0, std::shared_ptr<const BareGreenFunction> G0, const double U, const double K,
                        std::shared_ptr<GreenFunction> G);
    
    double solve();
    
    double fermiSign() const {return m_measuredfermisign;}
    
    double aveVertexOrder() const {return m_avevertorder;}
    
    const ArrayXindex &vertexOrderHistogram() const {return m_histogram;}
    
    void updateInterPhysParams();
};


#endif /* ct_aux_imp_solver_hpp */
