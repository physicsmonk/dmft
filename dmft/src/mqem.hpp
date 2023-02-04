//
//  mqem.hpp
//  dmft
//
//  Created by Yin Shi on 1/4/23.
//

#ifndef mqem_hpp
#define mqem_hpp

#include <cmath>
#include <map>
#include <any>
#include <utility>
#include <Eigen/Eigenvalues>
#include "gf_data_wrapper.hpp"

using namespace std::complex_literals;

template <int _n0, int _n1, int _nm>
class MQEMContinuator {
public:
    //static constexpr int n01 = _n0 == Eigen::Dynamic ? Eigen::Dynamic : _n0 + 1;
    std::map<std::string, std::any> parameters;
    
    MQEMContinuator() {initParams();}
    MQEMContinuator(const MQEMContinuator&) =  default;
    MQEMContinuator(MQEMContinuator&&) = default;
    MQEMContinuator& operator=(const MQEMContinuator&) = default;
    MQEMContinuator& operator=(MQEMContinuator&&) = default;
    
    template <int n_mom>
    MQEMContinuator(const Eigen::Array<double, _n1, 1>& mats_freq, const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                    const Eigen::Array<double, _n1, _n0>& Gwvar, const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom,
                    const std::size_t Nul, const double omegal, const std::size_t Nw, const double omegar, const std::size_t Nur) {
        initParams();
        assembleKernelMatrix(mats_freq, Nul, omegal, Nw, omegar, Nur);
        computeSpectra(mats_freq, Gw, Gwvar, mom, Nul, omegal, Nw, omegar, Nur);
    }
    
    // Useful for obtaining Matsubara function from spectral function
    void assembleKernelMatrix(const Eigen::Array<double, _n1, 1>& mats_freq, const std::size_t Nul, const double omegal, const std::size_t Nw, const double omegar,
                              const std::size_t Nur);
    template <int n_mom>
    bool computeSpectra(const Eigen::Array<double, _n1, 1>& mats_freq, const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                        const SqMatArray<double, _n0, _n1, _nm>& Gwvar, const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom);
    void computeRetardedFunc();
    void computeRetardedFunc(const SqMatArray<std::complex<double>, _n0, 1, _nm>& static_part);
    const Eigen::ArrayXd& realFreqGrid() const {return m_omega;}
    const SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm>& spectra() const {return m_A;}
    const SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm>& retardedFunc() const {return m_G_retarded;}
    const SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm>& defaultModel() const {return m_D;}
    // Useful for obtaining Matsubara function from spectral function at outside
    const Eigen::Matrix<std::complex<double>, _n1, Eigen::Dynamic>& kernelMatrix() const {return m_K;}
    // Useful for calculating principle integral from spectral function at outside
    const Eigen::MatrixXd& prinIntKerMat() const {return m_Kp;}
    const Eigen::VectorXd& realFreqIntVector() const {return m_intA;}
    std::size_t optimalAlphaIndex(const std::size_t slocal) const {return m_opt_alpha_id(slocal);}
    double optimalLog10alpha(const std::size_t slocal) const {return m_misfit_curve[slocal](m_opt_alpha_id(slocal), 0);}
    const Eigen::ArrayX3d& diagnosis(const std::size_t slocal) const {return m_misfit_curve[slocal];}
    // Argument curvature will be filled with solution after execution; its const-ness will be cast away inside the function; see Eigen library manual
    template <typename Derived, typename OtherDerived>
    static void fitCurvature(const Eigen::DenseBase<Derived>& curve, const Eigen::DenseBase<OtherDerived>& curvature, const std::size_t n_fitpts = 5);
    
private:
    Eigen::Array<std::size_t, Eigen::Dynamic, 1> m_opt_alpha_id;
    Eigen::ArrayXd m_omega;
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_A;
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_D;
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_log_normD;
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_G_retarded;
    Eigen::Matrix<std::complex<double>, _n1, Eigen::Dynamic> m_K;
    Eigen::MatrixXd m_Kp;
    Eigen::VectorXd m_intA;
    std::vector<Eigen::ArrayX3d> m_misfit_curve;  // Use std::vector because each Eigen::ArrayX2d could have different length
    
    template <typename T>
    static T a_side_coeff(const double ub, const double ue, const T x, const double omega0) {
        const T tmp = x - omega0;
        if (std::abs(tmp) < 1e-10) return (ue * ue * ue - ub * ub * ub) / 3.0;
        return -(ue - ub) / (tmp * tmp) - (ue * ue - ub * ub) / (2.0 * tmp) - std::log((1.0 - ue * tmp) / (1.0 - ub * tmp)) / (tmp * tmp * tmp);
    }
    template <typename T>
    static T b_side_coeff(const double ub, const double ue, const T x, const double omega0) {
        const T tmp = x - omega0;
        if (std::abs(tmp) < 1e-10) return (ue * ue - ub * ub) / 2.0;
        return -(ue - ub) / tmp - std::log((1.0 - ue * tmp) / (1.0 - ub * tmp)) / (tmp * tmp);
    }
    template <typename T>
    static T c_side_coeff(const double ub, const double ue, const T x, const double omega0) {
        const T tmp = x - omega0;
        if (std::abs(tmp) < 1e-10) return ue - ub;
        return -std::log((1.0 - ue * tmp) / (1.0 - ub * tmp)) / tmp;
    }
    template <typename T>
    static T d_side_coeff(const double ub, const double ue, const T x, const double omega0) {
        const T tmp = x - omega0;
        if (std::abs(tmp) < 1e-10) return std::log(ue / ub);
        return -std::log((1.0 - ue * tmp) * ub / ((1.0 - ub * tmp) * ue));
    }
    template <typename T>
    static T a_mid_coeff(const double omegab, const double omegae, const T x) {
        const T tmp = x - omegab;
        const double dw = omegae - omegab;
        return -tmp * tmp * dw - tmp * dw * dw / 2.0 - dw * dw * dw / 3.0 - tmp * tmp * tmp * std::log((omegae - x) / (omegab - x));
    }
    template <typename T>
    static T b_mid_coeff(const double omegab, const double omegae, const T x) {
        const T tmp = x - omegab;
        const double dw = omegae - omegab;
        return -tmp * dw - dw * dw / 2.0 - tmp * tmp * std::log((omegae - x) / (omegab - x));
    }
    template <typename T>
    static T c_mid_coeff(const double omegab, const double omegae, const T x) {
        const T tmp = x - omegab;
        const double dw = omegae - omegab;
        return -dw - tmp * std::log((omegae - x) / (omegab - x));
    }
    template <typename T>
    static T d_mid_coeff(const double omegab, const double omegae, const T x) {
        return -std::log((omegae - x) / (omegab - x));
    }
    
    void initParams() {
        parameters["principle_int_eps"] = 0.0001;
        parameters["Pulay_mixing_param"] = 0.005;
        parameters["Pulay_history_size"] = std::size_t(5);
        parameters["Pulay_period"] = std::size_t(3);   // Larger period typically leads to more stable but slowly converging iteractions
        parameters["Pulay_tolerance"] = 1e-5;
        parameters["Pulay_max_iteration"] = std::size_t(500);
        //parameters["Pulay_exp_limit"] = 300.0;
        parameters["Gaussian_sigma"] = 1.5;
        parameters["alpha_max_fac"] = 10.0;
        parameters["alpha_info_fit_fac"] = 0.05;
        parameters["alpha_init_fraction"] = 0.01;
        parameters["alpha_max_trial"] = std::size_t(30);
        parameters["alpha_stop_slope"] = 0.01;
        parameters["alpha_stop_step"] = 1e-5;
        parameters["alpha_spec_rel_err"] = 0.1;
        parameters["alpha_step_min_ratio"] = 0.5;
        parameters["alpha_step_max_ratio"] = 2.0;
        parameters["alpha_step_scale"] = 0.95;
        parameters["alpha_capacity"] = std::size_t(1000);
        parameters["alpha_curvature_fit_size"] = std::size_t(5);
        parameters["verbose"] = true;
    }
    template <int n_mom>
    void computeDefaultModel(const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom);
    void fixedPointRelation(const Eigen::Array<double, _n1, 1>& mats_freq, const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                            const SqMatArray<double, _n0, _n1, _nm>& Gwvar, const double m0trace, const double alpha,
                            const std::size_t s, SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm>& g) const;
    template <int n_mom>
    std::pair<bool, std::size_t> periodicPulaySolve(const Eigen::Array<double, _n1, 1>& mats_freq, const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                                                    const SqMatArray<double, _n0, _n1, _nm>& Gwvar, const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom,
                                                    const double alpha, const std::size_t s);
    template <typename Scalar, int n0, int n1, int nm>
    double normInt(const SqMatArray<Scalar, n0, n1, nm>& integrand) const;
    // Calculate chi^2
    double misfit(const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw, const SqMatArray<double, _n0, _n1, _nm>& Gwvar, const std::size_t s) const;
};

// Must call assembleKernelMatrix first
template <int _n0, int _n1, int _nm>
template <int n_mom>
bool MQEMContinuator<_n0, _n1, _nm>::computeSpectra(const Eigen::Array<double, _n1, 1>& mats_freq,
                                                    const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                                                    const SqMatArray<double, _n0, _n1, _nm>& Gwvar,
                                                    const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom) {
    const auto amaxfac = std::any_cast<double>(parameters.at("alpha_max_fac"));
    const auto ainfofitfac = std::any_cast<double>(parameters.at("alpha_info_fit_fac"));
    const auto ainitfrac = std::any_cast<double>(parameters.at("alpha_init_fraction"));
    const auto amaxtrial = std::any_cast<std::size_t>(parameters.at("alpha_max_trial"));
    const auto astopslope = std::any_cast<double>(parameters.at("alpha_stop_slope"));
    const auto astopstep = std::any_cast<double>(parameters.at("alpha_stop_step"));
    const auto verbose = std::any_cast<bool>(parameters.at("verbose"));
    const auto dAtol = std::any_cast<double>(parameters.at("alpha_spec_rel_err"));
    const auto rmin = std::any_cast<double>(parameters.at("alpha_step_min_ratio"));
    const auto rmax = std::any_cast<double>(parameters.at("alpha_step_max_ratio"));
    const auto sa = std::any_cast<double>(parameters.at("alpha_step_scale"));
    const auto acapacity = std::any_cast<std::size_t>(parameters.at("alpha_capacity"));
    const auto afitsize = std::any_cast<std::size_t>(parameters.at("alpha_curvature_fit_size"));
    const double eps = 1e-10;
    if (amaxfac < ainfofitfac) throw std::range_error("computeSpectra: alpha_max_fac should not be smaller than alpha_info_fit_fac");
    if (amaxtrial < 1) throw std::range_error("computeSpectra: num_alpha cannot be less than 1");
    
    double varmin, logainfofit, loga, logchi2, logchi2_old, dloga, dloga_fac, dA, slope;
    std::size_t na, trial;
    std::pair<bool, std::size_t> cvg;
    
    m_D.mpiCommunicator(Gw.mpiCommunicator());
    m_log_normD.mpiCommunicator(Gw.mpiCommunicator());
    computeDefaultModel(mom);  // m_D, m_log_normD allocated and calculated in here
    
    m_A.mpiCommunicator(Gw.mpiCommunicator());
    m_A.resize(Gw.dim0(), m_omega.size(), Gw.dimm());
    
    const auto Gwpart = Gw.mastDim0Part();
    const auto Gwvarpart = Gwvar.mastDim0Part();
    auto Apart = m_A.mastDim0Part();
    auto Dpart = m_D.mastDim0Part();
    std::vector<SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm> > As(acapacity);
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm> A_old(1, m_A.dim1(), m_A.dimm());
    //Eigen::ArrayXd curvature, deriv, ainterval;
    bool converged = true;
    m_opt_alpha_id.resize(Gwpart.dim0());
    m_misfit_curve.resize(Gwpart.dim0());
    Apart() = Dpart();  // Initialize m_A
    if (verbose && Gw.processRank() == 0) std::cout << std::scientific << std::setprecision(3)
        << "====== MQEM: start decreasing alpha in process 0 ======" << std::endl;
    for (std::size_t s = 0; s < Gwpart.dim0(); ++s) {
        // Initialize quantities
        na = 0;
        trial = 0;
        dloga = 0.0;  // Set zero for initial calculation
        slope = 10.0 * std::abs(astopslope);
        varmin = Gwvarpart.atDim0(s).minCoeff();
        loga = std::log10(amaxfac / varmin);
        logainfofit = std::log10(ainfofitfac / varmin);
        cvg.first = true;
        m_opt_alpha_id(s) = 0;
        logchi2 = std::log10(misfit(Gw, Gwvar, s + Gwpart.start()));
        m_misfit_curve[s].resize(acapacity, Eigen::NoChange);
        if (verbose && Gw.processRank() == 0) std::cout << "------ Spin " << s + Gwpart.start() << " ------" << std::endl
            << "    stepID log10alpha log10chi^2   stepSize      slope #PulayIter #PulayFail" << std::endl;
        do {
            if (trial > amaxtrial) {
                converged = false;
                std::cout << "MQEM computeSpectra: maximum number of trials reached (diverged) at dim0 " << s + Gwpart.start() << std::endl;
                break;
            }
            if (cvg.first) A_old() = Apart.atDim0(s);
            try {
                cvg = periodicPulaySolve(mats_freq, Gw, Gwvar, mom, std::pow(10.0, loga - dloga), s + Gwpart.start());  // m_A updated in here
            }
            catch (const std::runtime_error& e) {
                cvg.first = false;
                std::cout << "  Uninvertible matrix encountered in periodic Pulay solver" << std::endl;
            }
            if (!cvg.first) {  // Solve diverged
                if (na == 0) {  // Initial solve
                    converged = false;
                    std::cout << "MQEM computeSpectra: solving for initial alpha diverged at dim0 " << s + Gwpart.start() << std::endl;
                    break;
                }
                Apart.atDim0(s) = A_old();  // Restore initial guess
                dloga *= rmin;
                //parameters.at("Pulay_period") = std::any_cast<std::size_t>(parameters.at("Pulay_period")) + 1;
                parameters.at("Pulay_mixing_param") = std::any_cast<double>(parameters.at("Pulay_mixing_param")) * rmin;
                ++trial;
                continue;
            }
            // Solve converged
            loga -= dloga;
            logchi2_old = logchi2;
            logchi2 = std::log10(misfit(Gw, Gwvar, s + Gwpart.start()));
            // Initial slope set to zero to pretend that initial alpha is large enough for spectrum to be close enough to default model
            slope = na == 0 ? 0.0 : (logchi2_old - logchi2) / dloga;
            
            As[na].resize(1, m_A.dim1(), m_A.dimm());
            As[na]() = Apart.atDim0(s);
            m_misfit_curve[s](na, 0) = loga;
            m_misfit_curve[s](na, 1) = logchi2;
            
            if (verbose && Gw.processRank() == 0) std::cout << std::setw(10) << na << " " << std::setw(10) << loga << " " << std::setw(10) << logchi2
                << " " << std::setw(10) << dloga << " " << std::setw(10) << slope << " " << std::setw(10) << cvg.second
                << " " << std::setw(10) << trial << std::endl;
            
            // Update step size
            if (na == 0) dloga = (loga - logainfofit) * ainitfrac;  // Really initialize step size here
            else {
                dA = std::sqrt((Apart.atDim0(s) - A_old()).cwiseAbs2().sum() / A_old().cwiseAbs2().sum());
                //Adiff() = Apart.atDim0(s) - A_old();
                //dA = normInt(Adiff);
                //if (dA < dAtol) parameters.at("Pulay_period") = std::max(std::any_cast<std::size_t>(parameters.at("Pulay_period")) - 1, std::size_t(2));
                dloga_fac = std::min(std::max(sa * std::sqrt(dAtol / std::max(dA, eps)), rmin), rmax);
                dloga *= dloga_fac;
                parameters.at("Pulay_mixing_param") = std::any_cast<double>(parameters.at("Pulay_mixing_param")) * dloga_fac;
            }
            
            ++na;
            trial = 0;
        } while ((slope > astopslope || loga > logainfofit) && dloga > astopstep && na < acapacity);
        if (verbose && Gw.processRank() == 0) {
            std::cout << "------ Spin " << s + Gwpart.start() << ": stopped due to ";
            if (slope <= astopslope) std::cout << "small slope ";
            else if (dloga <= astopstep) std::cout << "small step ";
            else if (na >= acapacity) std::cout << "full reservoir ";
            else std::cout << "divergence ";
            std::cout << "------" << std::endl;
        }
        m_misfit_curve[s].conservativeResize(na, Eigen::NoChange);  // Get size right
        if (na > 2) {  // Calculate misfit curve curvature and find optimal alpha and spectrum
            //m_misfit_curve[s](0, 2) = std::nan("initial");
            //m_misfit_curve[s](1, 2) = std::nan("initial");
            //ainterval = m_misfit_curve[s](Eigen::seq(0, na - 2), 0) - m_misfit_curve[s](Eigen::seq(1, na - 1), 0);
            //deriv = (m_misfit_curve[s](Eigen::seq(0, na - 2), 1) - m_misfit_curve[s](Eigen::seq(1, na - 1), 1)) / ainterval;  // Forward first-order derivative
            //m_misfit_curve[s](Eigen::seq(2, na - 1), 2) = (deriv.head(na - 2) - deriv.tail(na - 2)) / ainterval.tail(na - 2);  // Forward second-order derivative
            //m_misfit_curve[s](Eigen::seq(2, na - 1), 2) /= (1.0 + deriv.tail(na - 2).square()).cube().sqrt();  // Signed curvature
            //m_misfit_curve[s](Eigen::seq(2, na - 1), 2).maxCoeff(&(m_opt_alpha_id(s)));
            //m_opt_alpha_id(s) += 2;
            fitCurvature(m_misfit_curve[s].template leftCols<2>(), m_misfit_curve[s].col(2), afitsize);
            m_misfit_curve[s].col(2).template maxCoeff<Eigen::PropagateNumbers>(&(m_opt_alpha_id(s)));
            Apart.atDim0(s) = As[m_opt_alpha_id(s)]();
        }
        else std::cout << "MQEM computeSpectra: cannot determine optimal alpha because misfit curve has less than 3 points" << std::endl;
    }
    if (verbose && Gw.processRank() == 0) std::cout << "====== MQEM: end decreasing alpha in process 0 ======" << std::endl;
    Apart.allGather();
    return converged;  // Each process return its own convergence status
}

template <int _n0, int _n1, int _nm>
void MQEMContinuator<_n0, _n1, _nm>::computeRetardedFunc() {
    m_G_retarded.mpiCommunicator(m_A.mpiCommunicator());
    m_G_retarded.resize(m_A.dim0(), m_A.dim1(), m_A.dimm());
    auto GRpart = m_G_retarded.mastDim0Part();
    auto Apart = m_A.mastDim0Part();
    GRpart() = -1i * M_PI * Apart();
    for (std::size_t s = 0; s < Apart.dim0(); ++s) {
        GRpart.dim1RowVecsAtDim0(s).transpose().noalias() += m_Kp * Apart.dim1RowVecsAtDim0(s).transpose();  // Principle integral
    }
    GRpart.allGather();
}

template <int _n0, int _n1, int _nm>
void MQEMContinuator<_n0, _n1, _nm>::computeRetardedFunc(const SqMatArray<std::complex<double>, _n0, 1, _nm> &static_part) {
    computeRetardedFunc();
    for (std::size_t i = 0; i < m_G_retarded.dim1(); ++i) m_G_retarded.atDim1(i) += static_part();
}

template <int _n0, int _n1, int _nm>
void MQEMContinuator<_n0, _n1, _nm>::assembleKernelMatrix(const Eigen::Array<double, _n1, 1> &mats_freq, const std::size_t Nul, const double omegal,
                                                          const std::size_t Nw, const double omegar, const std::size_t Nur) {
    if (Nul < 1) throw std::range_error("computeKernelMatrix: Nul cannot be less than 1");
    if (Nw < 2) throw std::range_error("computeKernelMatrix: Nw cannot be less than 2");
    if (Nur < 1) throw std::range_error("computeKernelMatrix: Nur cannot be less than 1");
    
    const auto eps = std::any_cast<double>(parameters.at("principle_int_eps"));
    const double domega = (omegar - omegal) / (Nw - 1);
    const double omega0l = omegal + Nul * domega;
    const double dul = 1.0 / (Nul * (omegal - domega - omega0l));  // This is negative; point sequence dul, 2*dul, ..., Nul*dul corresponds omega from left to right
    const double omega0r = omegar - Nur * domega;
    const double dur = 1.0 / (Nur * (omegar + domega - omega0r));  // This is positive; point sequence dur, 2*dur, ..., Nur*dur corresponds omega from right to left
    const double ur = (Nur + 1) * dur;
    
    const std::size_t n_iomega = mats_freq.size();
    const std::size_t Nspl = Nul + Nw - 1 + Nur;
    const std::size_t Ncoeff = 4 * Nspl;
    // Matrix connecting Matsubara Green's function and cubic spline coefficients of spectral function
    Eigen::Matrix<std::complex<double>, _n1, Eigen::Dynamic> K(n_iomega, Ncoeff);  // Every element of K will be explicitly set, so no need of initialization
    Eigen::MatrixXd Kp(Nspl + 1, Ncoeff);
    Eigen::VectorXd intA(Ncoeff);  // Every element will be explicitly set, so no need of initialization
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(Ncoeff, Ncoeff), T = Eigen::MatrixXd::Zero(Ncoeff, Nspl + 1);
    double uj, uj1, ujp, uj1p, omegajp, omegaj1p;
    std::size_t ja, jglobal;
    m_omega.resize(Nspl + 1);
    
    // First fill real frequencies; all the real frequencies will be needed later
    for (std::size_t j = 0; j < Nul; ++j) m_omega(j) = 1.0 / ((j + 1) * dul) + omega0l;
    for (std::size_t j = 0; j < Nw; ++j) m_omega(j + Nul) = j * domega + omegal;
    for (std::size_t j = 0; j < Nur; ++j) m_omega(j + Nul + Nw) = 1.0 / ((Nur - j) * dur) + omega0r;
    
    // For left side
    for (std::size_t j = 0; j < Nul; ++j) {  // The number of spline is Nul for the left side
        ja = 4 * j;
        uj = (j + 1) * dul;
        uj1 = (j + 2) * dul;
        //m_omega(j) = 1.0 / uj + omega0l;
        for (std::size_t n = 0; n < n_iomega; ++n) {
            K(n, ja) = a_side_coeff(uj, uj1, 1i * mats_freq(n), omega0l);
            K(n, ja + 1) = b_side_coeff(uj, uj1, 1i * mats_freq(n), omega0l);
            K(n, ja + 2) = c_side_coeff(uj, uj1, 1i * mats_freq(n), omega0l);
            K(n, ja + 3) = d_side_coeff(uj, uj1, 1i * mats_freq(n), omega0l);
        }
        for (std::size_t i = 0; i < Nspl + 1; ++i) {
            if (i == j + 1) {  // Spline immediately left to point i
                ujp = uj;
                uj1p = uj1 / (1.0 - eps * uj1);
            }
            else if (i == j) {  // Spline immediately right to point i
                ujp = uj / (1.0 + eps * uj);
                uj1p = uj1;
            }
            else {
                ujp = uj;
                uj1p = uj1;
            }
            Kp(i, ja) = a_side_coeff(ujp, uj1p, m_omega(i), omega0l);
            Kp(i, ja + 1) = b_side_coeff(ujp, uj1p, m_omega(i), omega0l);
            Kp(i, ja + 2) = c_side_coeff(ujp, uj1p, m_omega(i), omega0l);
            Kp(i, ja + 3) = d_side_coeff(ujp, uj1p, m_omega(i), omega0l);
        }
        intA(ja) = -(uj1 * uj1 - uj * uj) / 2.0; intA(ja + 1) = -(uj1 - uj); intA(ja + 2) = -std::log(uj1 / uj); intA(ja + 3) = 1.0 / uj1 - 1.0 / uj;
        B(ja, ja) = uj * uj * uj; B(ja, ja + 1) = uj * uj; B(ja, ja + 2) = uj; B(ja, ja + 3) = 1.0;
        T(ja, j) = 1.0;
        B(ja + 1, ja) = uj1 * uj1 * uj1; B(ja + 1, ja + 1) = uj1 * uj1; B(ja + 1, ja + 2) = uj1; B(ja + 1, ja + 3) = 1.0;
        T(ja + 1, j + 1) = 1.0;
        if (j < Nul - 1) {
            B(ja + 2, ja) = 3.0 * uj1 * uj1; B(ja + 2, ja + 1) = 2.0 * uj1; B(ja + 2, ja + 2) = 1.0;
            B(ja + 2, ja + 4) = -3.0 * uj1 * uj1; B(ja + 2, ja + 5) = -2.0 * uj1; B(ja + 2, ja + 6) = -1.0;
            B(ja + 3, ja) = 6.0 * uj1; B(ja + 3, ja + 1) = 2.0;
            B(ja + 3, ja + 4) = -6.0 * uj1; B(ja + 3, ja + 5) = -2.0;
        }
    }
    
    // For derivatives at omegal; ja and uj1 just have the proper values now
    B(ja + 2, ja) = -uj1 * uj1 * 3.0 * uj1 * uj1; B(ja + 2, ja + 1) = -uj1 * uj1 * 2.0 * uj1; B(ja + 2, ja + 2) = -uj1 * uj1;
    B(ja + 2, ja + 6) = -1.0;  // Arising from the first spline in the middle part
    B(ja + 3, ja) = 12.0 * uj1 * uj1 * uj1 * uj1 * uj1; B(ja + 3, ja + 1) = 6.0 * uj1 * uj1 * uj1 * uj1; B(ja + 3, ja + 2) = 2.0 * uj1 * uj1 * uj1;
    B(ja + 3, ja + 5) = -2.0;  // Arising from the first spline in the middle part
    
    // For middle part
    for (std::size_t j = 0; j < Nw - 1; ++j) {  // The number of spline is Nw - 1 for the middle part
        jglobal = j + Nul;
        ja = 4 * jglobal;
        //omegaj = j * domega + omegal;
        //m_omega(jglobal) = omegaj;
        for (std::size_t n = 0; n < n_iomega; ++n) {
            K(n, ja) = a_mid_coeff(m_omega(jglobal), m_omega(jglobal) + domega, 1i * mats_freq(n));
            K(n, ja + 1) = b_mid_coeff(m_omega(jglobal), m_omega(jglobal) + domega, 1i * mats_freq(n));
            K(n, ja + 2) = c_mid_coeff(m_omega(jglobal), m_omega(jglobal) + domega, 1i * mats_freq(n));
            K(n, ja + 3) = d_mid_coeff(m_omega(jglobal), m_omega(jglobal) + domega, 1i * mats_freq(n));
        }
        for (std::size_t i = 0; i < Nspl + 1; ++i) {
            if (i == jglobal + 1) {  // Spline immediately left to point i
                omegajp = m_omega(jglobal);
                omegaj1p = m_omega(jglobal) + domega - eps;
            }
            else if (i == jglobal) {  // Spline immediately right to point i
                omegajp = m_omega(jglobal) + eps;
                omegaj1p = m_omega(jglobal) + domega;
            }
            else {
                omegajp = m_omega(jglobal);
                omegaj1p = m_omega(jglobal) + domega;
            }
            Kp(i, ja) = a_mid_coeff(omegajp, omegaj1p, m_omega(i));
            Kp(i, ja + 1) = b_mid_coeff(omegajp, omegaj1p, m_omega(i));
            Kp(i, ja + 2) = c_mid_coeff(omegajp, omegaj1p, m_omega(i));
            Kp(i, ja + 3) = d_mid_coeff(omegajp, omegaj1p, m_omega(i));
        }
        intA(ja) = domega * domega * domega * domega / 4.0; intA(ja + 1) = domega * domega * domega / 3.0; intA(ja + 2) = domega * domega / 2.0; intA(ja + 3) = domega;
        B(ja, ja + 3) = 1.0;
        T(ja, jglobal) = 1.0;
        B(ja + 1, ja) = domega * domega * domega; B(ja + 1, ja + 1) = domega * domega; B(ja + 1, ja + 2) = domega; B(ja + 1, ja + 3) = 1.0;
        T(ja + 1, jglobal + 1) = 1.0;
        if (j < Nw - 2) {
            B(ja + 2, ja) = 3.0 * domega * domega; B(ja + 2, ja + 1) = 2.0 * domega; B(ja + 2, ja + 2) = 1.0;
            B(ja + 2, ja + 6) = -1.0;
            B(ja + 3, ja) = 6.0 * domega; B(ja + 3, ja + 1) = 2.0;
            B(ja + 3, ja + 5) = -2.0;
        }
    }
    //m_omega(jglobal + 1) = omegar;
    
    // For derivatives at omegar; ja just has the proper value now
    B(ja + 2, ja) = 3.0 * domega * domega; B(ja + 2, ja + 1) = 2.0 * domega; B(ja + 2, ja + 2) = 1.0;
    // Arising from the first spline in the right side
    B(ja + 2, ja + 4) = ur * ur * 3.0 * ur * ur; B(ja + 2, ja + 5) = ur * ur * 2.0 * ur; B(ja + 2, ja + 6) = ur * ur;
    B(ja + 3, ja) = 6.0 * domega; B(ja + 3, ja + 1) = 2.0;
    // Arising from the first spline in the right side
    B(ja + 3, ja + 4) = -12.0 * ur * ur * ur * ur * ur; B(ja + 3, ja + 5) = -6.0 * ur * ur * ur * ur; B(ja + 3, ja + 6) = -2.0 * ur * ur * ur;
    
    // For right side
    for (std::size_t j = 0; j < Nur; ++j) {
        jglobal = j + Nul + Nw - 1;
        ja = 4 * jglobal;
        uj = (Nur - j + 1) * dur;
        uj1 = (Nur - j) * dur;
        //m_omega(jglobal + 1) = 1.0 / uj1 + omega0r;
        for (std::size_t n = 0; n < n_iomega; ++n) {
            K(n, ja) = a_side_coeff(uj, uj1, 1i * mats_freq(n), omega0r);
            K(n, ja + 1) = b_side_coeff(uj, uj1, 1i * mats_freq(n), omega0r);
            K(n, ja + 2) = c_side_coeff(uj, uj1, 1i * mats_freq(n), omega0r);
            K(n, ja + 3) = d_side_coeff(uj, uj1, 1i * mats_freq(n), omega0r);
        }
        for (std::size_t i = 0; i < Nspl + 1; ++i) {
            if (i == jglobal + 1) {  // Spline immediately left to point i
                ujp = uj;
                uj1p = uj1 / (1.0 - eps * uj1);
            }
            else if (i == jglobal) {  // Spline immediately right to point i
                ujp = uj / (1.0 + eps * uj);
                uj1p = uj1;
            }
            else {
                ujp = uj;
                uj1p = uj1;
            }
            Kp(i, ja) = a_side_coeff(ujp, uj1p, m_omega(i), omega0r);
            Kp(i, ja + 1) = b_side_coeff(ujp, uj1p, m_omega(i), omega0r);
            Kp(i, ja + 2) = c_side_coeff(ujp, uj1p, m_omega(i), omega0r);
            Kp(i, ja + 3) = d_side_coeff(ujp, uj1p, m_omega(i), omega0r);
        }
        intA(ja) = -(uj1 * uj1 - uj * uj) / 2.0; intA(ja + 1) = -(uj1 - uj); intA(ja + 2) = -std::log(uj1 / uj); intA(ja + 3) = 1.0 / uj1 - 1.0 / uj;
        B(ja, ja) = uj * uj * uj; B(ja, ja + 1) = uj * uj; B(ja, ja + 2) = uj; B(ja, ja + 3) = 1.0;
        T(ja, jglobal) = 1.0;
        B(ja + 1, ja) = uj1 * uj1 * uj1; B(ja + 1, ja + 1) = uj1 * uj1; B(ja + 1, ja + 2) = uj1; B(ja + 1, ja + 3) = 1.0;
        T(ja + 1, jglobal + 1) = 1.0;
        if (j < Nur - 1) {
            B(ja + 2, ja) = 3.0 * uj1 * uj1; B(ja + 2, ja + 1) = 2.0 * uj1; B(ja + 2, ja + 2) = 1.0;
            B(ja + 2, ja + 4) = -3.0 * uj1 * uj1; B(ja + 2, ja + 5) = -2.0 * uj1; B(ja + 2, ja + 6) = -1.0;
            B(ja + 3, ja) = 6.0 * uj1; B(ja + 3, ja + 1) = 2.0;
            B(ja + 3, ja + 4) = -6.0 * uj1; B(ja + 3, ja + 5) = -2.0;
        }
    }
    
    // Boundary conditions
    B(Ncoeff - 2, 2) = 1.0;
    B(Ncoeff - 1, Ncoeff - 2) = 1.0;
    
    // Finally calculate the full kernel matrix
    //Eigen::MatrixXd Binv(Ncoeff, Ncoeff);
    //Binv = B.partialPivLu().inverse();
    //std::cout << K.array().isNaN().any() << ", " << Binv.array().isNaN().any() << std::endl;
    //m_K.resize(n_iomega, Nspl + 1);
    Eigen::MatrixXd tmp;
    tmp.noalias() = B.partialPivLu().inverse() * T;
    m_K.noalias() = K * tmp;
    m_Kp.noalias() = Kp * tmp;
    //std::size_t nanrow, nancol;
    //std::cout << Kp.array().isNaN().maxCoeff(&nanrow, &nancol) << std::endl;
    //std::cout << nanrow << ", " << nancol << std::endl;
    m_intA.transpose().noalias() = intA.transpose() * tmp;
}

// Currently only use the first (zero-th) moment; MPI communicators of m_D and m_log_normD must be set before
template <int _n0, int _n1, int _nm>
template <int n_mom>
void MQEMContinuator<_n0, _n1, _nm>::computeDefaultModel(const SqMatArray<std::complex<double>, _n0, n_mom, _nm> &mom) {
    const auto sigma = std::any_cast<double>(parameters.at("Gaussian_sigma"));
    double fac, m0trace;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<std::complex<double>, _nm, _nm> > es(mom.dimm());
    m_D.resize(mom.dim0(), m_omega.size(), mom.dimm());
    m_log_normD.resize(mom.dim0(), m_omega.size(), mom.dimm());
    auto Dpart = m_D.mastDim0Part();
    auto logDpart = m_log_normD.mastDim0Part();
    for (std::size_t s = 0; s < Dpart.dim0(); ++s) {
        m0trace = mom(s + Dpart.start(), 0).trace().real();  // Trace must be real
        // Diagonalize the default model; in the current version, only need to diagonalize the first moment; all moments are Hermitian
        es.compute(mom(s + Dpart.start(), 0));
        for (std::size_t j = 0; j < m_D.dim1(); ++j) {
            fac = std::exp(-m_omega(j) * m_omega(j) / (2.0 * sigma * sigma)) / sigma * M_SQRT1_2 * 0.5 * M_2_SQRTPI;
            Dpart(s, j) = mom(s + Dpart.start(), 0) * fac;
            logDpart(s, j) = es.eigenvectors() * (es.eigenvalues().array() * fac / m0trace).log().matrix().asDiagonal() * es.eigenvectors().adjoint();
        }
    }
    Dpart.allGather();
    logDpart.allGather();
}

// Renormalize quantities by m0trace in situ
template <int _n0, int _n1, int _nm>
void MQEMContinuator<_n0, _n1, _nm>::fixedPointRelation(const Eigen::Array<double, _n1, 1>& mats_freq,
                                                        const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                                                        const SqMatArray<double, _n0, _n1, _nm>& Gwvar,
                                                        double m0trace, const double alpha,
                                                        const std::size_t s,
                                                        SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm>& g) const {
    const std::size_t nm = Gw.dimm();
    const std::size_t n_iomega = Gw.dim1();
    const std::size_t n_omega = m_A.dim1();
    SqMatArray<std::complex<double>, 1, _n1, _nm> chi(1, n_iomega, nm);
    
    chi() = Gw.atDim0(s);
    chi.dim1RowVecsAtDim0(0).transpose().noalias() -= m_K * m_A.dim1RowVecsAtDim0(s).transpose();
    
    //Eigen::Matrix<double, _nm, Eigen::Dynamic> eigvals(nm, n_omega);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<std::complex<double>, _nm, _nm> > es(nm);
    Eigen::VectorXd gtr(n_omega);
    //Eigen::Array<double, _nm, 1> H1(nm);
    //double testmin = 0.0, testmax = 0.0;
    g.resize(1, n_omega, nm);
    g().setZero();
    for (std::size_t i = 0; i < n_omega; ++i) {
        for (std::size_t n = 0; n < n_iomega; ++n) g[i].noalias() += (m0trace * chi[n].array() / (2.0 * (1i * mats_freq(n) + m_omega(i)) * Gwvar(s, n).array() * alpha)).matrix();
        g[i] += g[i].adjoint().eval() - m_log_normD(s, i);  // Used eval() to explicitly evaluate to temporary so no aliasing issue
        es.compute(g[i]);
        //eigvals.col(i) = -es.eigenvalues();
        //g[i] = es.eigenvectors();
        g[i].noalias() = es.eigenvectors() * (-es.eigenvalues().array()).exp().matrix().asDiagonal() * es.eigenvectors().adjoint();
        gtr(i) = g[i].trace().real();  // Trace must be real
    }
    
    //const auto limexp = std::any_cast<double>(parameters.at("Pulay_exp_limit"));
    //double maxexp, offset;
    //maxexp = eigvals.maxCoeff();
    //offset = maxexp > limexp ? maxexp - limexp : 0.0;
    //for (std::size_t i = 0; i < n_omega; ++i) {
    //    g[i] = g[i] * (eigvals.col(i).array() - offset).exp().matrix().asDiagonal() * g[i].adjoint();
    //    gtr(i) = g[i].trace().real();  // Trace must be real
    //}
    double Z = m_intA.dot(gtr);  // Normalization factor
    g() *= m0trace / Z;
}

// Not initializing m_A, just taking whatever in m_A as the initial guess, which is advantageous for concecutive calculations as varying m_alpha smoothly
template <int _n0, int _n1, int _nm>
template <int n_mom>
std::pair<bool, std::size_t> MQEMContinuator<_n0, _n1, _nm>::periodicPulaySolve(const Eigen::Array<double, _n1, 1>& mats_freq,
                                                                                const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                                                                                const SqMatArray<double, _n0, _n1, _nm>& Gwvar,
                                                                                const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom,
                                                                                const double alpha,
                                                                                const std::size_t s) {
    const auto mix_param = std::any_cast<double>(parameters.at("Pulay_mixing_param"));
    const auto hist_size = std::any_cast<std::size_t>(parameters.at("Pulay_history_size"));
    const auto period = std::any_cast<std::size_t>(parameters.at("Pulay_period"));
    const auto tol = std::any_cast<double>(parameters.at("Pulay_tolerance"));
    const auto max_iter = std::any_cast<std::size_t>(parameters.at("Pulay_max_iteration"));
    if (hist_size == 0) throw std::range_error("periodicPulaySolve: history size cannot be zero");
    if (period == 0) throw std::range_error("periodicPulaySolve: period cannot be zero");
    
    const std::size_t n_omega = m_A.dim1();
    const std::size_t nm = m_A.dimm();
    const std::size_t N = n_omega * nm * nm;
    // No need to initialize; use these only after they are fully filled
    Eigen::MatrixXcd R(N, hist_size), R1(N, hist_size), F(N, hist_size), F1(N, hist_size);
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm> A_old(1, n_omega, nm), f(1, n_omega, nm), f_old(1, n_omega, nm);
    std::size_t iter, ih;
    double err = 1000.0;
    double m0trace;
    Eigen::PermutationMatrix<Eigen::Dynamic> perm(hist_size);
    Eigen::MatrixXcd FTF_inv(hist_size, hist_size);
    //Eigen::LLT<Eigen::MatrixXcd> llt(hist_size);
    //Eigen::PartialPivLU<Eigen::MatrixXcd> decomp(hist_size);
    //Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> decomp(hist_size, hist_size);
    Eigen::FullPivLU<Eigen::MatrixXcd> decomp(hist_size, hist_size);
    //Eigen::FullPivHouseholderQR<Eigen::MatrixXcd> decomp(hist_size, hist_size);
    //Eigen::JacobiSVD<Eigen::MatrixXcd, Eigen::NoQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeThinV> decomp(hist_size, hist_size);
    
    m0trace = mom(s, 0).trace().real();  // Trace must be a real number
    for (iter = 0; iter <= max_iter; ++iter) {
        // Calculate some current variables for computing new ones
        fixedPointRelation(mats_freq, Gw, Gwvar, m0trace, alpha, s, f);   // Read m_A in here
        f() -= m_A.atDim0(s);
        //err = std::sqrt(f().cwiseAbs2().sum() / n_omega);
        err = normInt(f);
        if (!std::isfinite(err)) return std::make_pair(false, iter);
        else if (err < tol) return std::make_pair(true, iter);
        //std::cout << "error = " << err << std::endl;
        // Record current variables into history caches, note the recording order is not proper in the caches
        if (iter > 0) {
            ih = (iter - 1) % hist_size;
            R1.col(ih) = (m_A.atDim0(s) - A_old()).reshaped();  // RHS flattened to a vector
            F1.col(ih) = (f() - f_old()).reshaped();
        }
        // Record current variables
        A_old() = m_A.atDim0(s);
        f_old() = f();
        // Calculate new variables
        if ((iter + 1) % period != 0 || iter < hist_size) m_A.atDim0(s) += mix_param * f();  // Linear mixing
        else {  // Pulay mixing, do this only after history caches are full
            // Prepare permutation matrix to properly order the history
            for (std::size_t ip = 0; ip < hist_size; ++ip) perm.indices()[ip] = (ip + 1 + ih) % hist_size;
            // Correctly reorder history, i.e., latest to the most right column; use new variables so not to interrupt the history caches
            R.noalias() = R1 * perm;
            F.noalias() = F1 * perm;
            //FTF_inv.noalias() = F.adjoint() * F;
            //llt.compute(FTF_inv);
            //FTF_inv = llt.solve(Eigen::MatrixXcd::Identity(hist_size, hist_size));  // Inverse of FTF
            //m_A.atDim0(s).noalias() += mix_param * f() - ((R + mix_param * F) * FTF_inv * F.adjoint() * f().reshaped()).reshaped(nm, nm * n_omega);
            FTF_inv.noalias() = F.transpose() * F;
            decomp.compute(FTF_inv);
            //FTF_inv = decomp.solve(Eigen::MatrixXcd::Identity(hist_size, hist_size));  // Inverse of FTF
            if (decomp.isInvertible()) FTF_inv = decomp.inverse();
            else throw std::runtime_error("Periodic Pulay solver: F^T * F not invertible");
            m_A.atDim0(s).noalias() += mix_param * f() - ((R + mix_param * F) * FTF_inv * F.transpose() * f().reshaped()).reshaped(nm, nm * n_omega);
        }
    }
    return std::make_pair(false, iter - 1);
}

template <int _n0, int _n1, int _nm>
template <typename Scalar, int n0, int n1, int nm>
double MQEMContinuator<_n0, _n1, _nm>::normInt(const SqMatArray<Scalar, n0, n1, nm>& integrand) const {
    assert(integrand.dim1() == m_intA.size());
    double result = 0.0;
    for (std::size_t s = 0; s < integrand.dim0(); ++s) result += m_intA.dot(integrand.dim1RowVecsAtDim0(s).cwiseAbs().colwise().sum());
    return result;
}

template <int _n0, int _n1, int _nm>
double MQEMContinuator<_n0, _n1, _nm>::misfit(const SqMatArray<std::complex<double>, _n0, _n1, _nm> &Gw, const SqMatArray<double, _n0, _n1, _nm> &Gwvar,
                                              const std::size_t s) const {
    double chi2 = 0.0;
    SqMatArray<std::complex<double>, 1, _n1, _nm> dG(1, Gw.dim1(), Gw.dimm());
    dG() = Gw.atDim0(s);
    dG.dim1RowVecsAtDim0(0).transpose().noalias() -= m_K * m_A.dim1RowVecsAtDim0(s).transpose();
    for (std::size_t n = 0; n < Gw.dim1(); ++n) chi2 += (dG[n].array() / Gwvar(s, n).array().sqrt()).abs2().sum();
    return chi2;
}

template <int _n0, int _n1, int _nm>
template <typename Derived, typename OtherDerived>
void MQEMContinuator<_n0, _n1, _nm>::fitCurvature(const Eigen::DenseBase<Derived> &curve, const Eigen::DenseBase<OtherDerived> &curvature,
                                                  const std::size_t n_fitpts) {
    if (curve.rows() < n_fitpts) throw std::range_error("MQEMContinuator::fitCurvature: #points of input curve cannot be less than #local points to fit");
    if (n_fitpts < 3) throw std::range_error("MQEMContinuator::fitCurvature: #local points to fit cannot be less than 3");
    
    typedef typename Derived::Scalar Scalar;
    //typedef typename Eigen::internal::plain_col_type<Derived>::type VectorType;
    
    // A general circle has 3 parameters; 3 for #row for contiguous memory for local param matrix
    Eigen::Matrix<Scalar, 3, Derived::RowsAtCompileTime> param_mat(3, curve.rows());
    Eigen::Vector<Scalar, Derived::RowsAtCompileTime> param_vec(curve.rows());
    
    // Assemble the full param matrix and vector
    for (std::size_t i = 0; i < curve.rows(); ++i) {
        param_mat(0, i) = curve(i, 0);
        param_mat(1, i) = curve(i, 1);
        param_mat(2, i) = 1.0;
        param_vec(i) = -(curve(i, 0) * curve(i, 0) + curve(i, 1) * curve(i, 1));
    }
    
    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<Scalar, Eigen::Dynamic, 3> > decomp(n_fitpts, 3);
    Eigen::Vector<Scalar, 3> circ_coeffs;
    Scalar dx, dy;
    const std::size_t ifitbegin = n_fitpts / 2;
    const std::size_t ifitend = curve.rows() - (n_fitpts + 1) / 2;
    
    Eigen::DenseBase<OtherDerived>& curvature_ = const_cast<Eigen::DenseBase<OtherDerived>&>(curvature);  // Make curvature writable by casting away its const-ness
    curvature_.derived().resize(curve.rows());  // Resize the derived object
    
    // Calculate curvature by least square fitting (solving)
    for (std::size_t i = 0; i < curve.rows(); ++i) {
        if (i < ifitbegin) curvature_(i) = std::nan("initial");
        else if (i > ifitend) curvature_(i) = std::nan("last");
        else {
            decomp.compute(param_mat.middleCols(i - ifitbegin, n_fitpts).transpose());
            circ_coeffs = decomp.solve(param_vec.segment(i - ifitbegin, n_fitpts));
            circ_coeffs(0) /= -2.0;  // Center coordinates of the circle
            circ_coeffs(1) /= -2.0;
            dx = curve(i + 1, 0) - curve(i, 0);
            dy = (curve(i + 1, 1) - curve(i, 1)) * std::copysign(1.0, dx);
            dx = std::abs(dx);
            curvature_(i) = std::copysign(1.0 / std::sqrt(circ_coeffs(0) * circ_coeffs(0) + circ_coeffs(1) * circ_coeffs(1) - circ_coeffs(2)),
                                          (curve(i, 0) - circ_coeffs(0)) * dy - (curve(i, 1) - circ_coeffs(1)) * dx);
        }
    }
}

typedef MQEMContinuator<2, Eigen::Dynamic, Eigen::Dynamic> MQEMContinuator2XX;

#endif /* mqem_hpp */
