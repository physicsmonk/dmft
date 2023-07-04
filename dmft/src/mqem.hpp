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
    
    template <int n_mom, typename Derived>
    MQEMContinuator(const Eigen::Array<double, _n1, 1>& mats_freq, const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                    const Eigen::Array<double, _n1, _n0>& Gwvar, const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom,
                    const Eigen::Index Nul, const Eigen::ArrayBase<Derived>& midrealfreq_anchors_steps, const Eigen::Index Nur) {
        initParams();
        assembleKernelMatrix(mats_freq, Nul, midRealFreqs(midrealfreq_anchors_steps), Nur);
        computeSpectra(mats_freq, Gw, Gwvar, mom);
    }
    
    template <typename Derived>
    static Eigen::ArrayXd midRealFreqs(const Eigen::ArrayBase<Derived>& anchors_steps);
    template <typename Derived>
    void assembleKernelMatrix(const Eigen::Array<double, _n1, 1>& mats_freq, const Eigen::Index Nul, const Eigen::ArrayBase<Derived>& real_freq_mid,
                              const Eigen::Index Nur);
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
    Eigen::Index optimalAlphaIndex(const Eigen::Index slocal) const {return m_opt_alpha_id(slocal);}
    double optimalLog10alpha(const Eigen::Index slocal) const {return m_misfit_curve[slocal](m_opt_alpha_id(slocal), 0);}
    const Eigen::ArrayX4d& diagnosis(const Eigen::Index slocal) const {return m_misfit_curve[slocal];}
    // Argument curvature will be filled with solution after execution; its const-ness will be cast away inside the function; see Eigen library manual
    template <typename Derived, typename OtherDerived>
    static void fitCurvature(const Eigen::DenseBase<Derived>& curve, const Eigen::DenseBase<OtherDerived>& curvature, const Eigen::Index n_fitpts = 5);
    template <typename Derived, typename OtherDerived>
    Eigen::Vector<typename Derived::Scalar, 4> fitFDFunc(const Eigen::DenseBase<Derived>& curve, const Eigen::DenseBase<OtherDerived>& fitted);
    template <typename Derived, typename OtherDerived>
    static void fitCubicSpline(const Eigen::DenseBase<Derived>& curve, const Eigen::DenseBase<OtherDerived>& deriv_curv);
    
private:
    static constexpr int nm2 = _nm == Eigen::Dynamic ? Eigen::Dynamic : _nm * _nm;
    double m_pulay_mix_param;   // Internal Pulay mixing parameter that can change
    Eigen::Array<Eigen::Index, Eigen::Dynamic, 1> m_opt_alpha_id;
    Eigen::ArrayXd m_omega;
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_A;
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_D;
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_log_normD;
    SqMatArray<std::complex<double>, _n0, Eigen::Dynamic, _nm> m_G_retarded;
    Eigen::Matrix<std::complex<double>, _n1, Eigen::Dynamic> m_K, m_K_raw_Hc;
    Eigen::MatrixXd m_Kp;
    Eigen::VectorXd m_intA, m_intwA, m_intw2A;
    std::vector<Eigen::ArrayX4d> m_misfit_curve;  // Use std::vector because each Eigen::ArrayX4d could have different length
    
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
        parameters["simple_default_model"] = false;
        parameters["secant_max_iter"] = Eigen::Index(30);
        parameters["secant_tol"] = 0.001;
        parameters["secant_damp"] = 0.5;
        parameters["matrix_invertible_threshold"] = -1.0;
        parameters["principle_int_eps"] = 0.0001;
        parameters["Pulay_mixing_param"] = 0.005;
        parameters["Pulay_history_size"] = Eigen::Index(5);
        parameters["Pulay_period"] = Eigen::Index(3);   // Larger period typically leads to more stable but slowly converging iteractions
        parameters["Pulay_tolerance"] = 1e-5;
        parameters["Pulay_max_iteration"] = Eigen::Index(500);
        //parameters["Pulay_exp_limit"] = 300.0;
        parameters["Gaussian_sigma"] = 1.5;
        parameters["Gaussian_shift"] = 0.0;
        parameters["alpha_max_fac"] = 10.0;
        parameters["alpha_info_fit_fac"] = 0.05;
        parameters["alpha_init_fraction"] = 0.01;
        parameters["alpha_max_trial"] = Eigen::Index(30);
        parameters["alpha_stop_slope"] = 0.01;
        parameters["alpha_stop_step"] = 1e-5;
        parameters["alpha_spec_rel_err"] = 0.1;
        parameters["alpha_step_min_ratio"] = 0.5;
        parameters["alpha_step_max_ratio"] = 2.0;
        parameters["alpha_step_scale"] = 0.95;
        parameters["alpha_capacity"] = Eigen::Index(1000);
        parameters["alpha_cache_all"] = true;
        //parameters["alpha_curvature_fit_size"] = Eigen::Index(5);
        parameters["verbose"] = true;
        parameters["FDfit_damp"] = 0.1;
        parameters["FDfit_tolerance"] = 1e-4;
        parameters["FDfit_max_iteration"] = Eigen::Index(500);
    }
    template <int n_mom>
    void momentConstraints(const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& moms, const Eigen::Index s, const SqMatArray<std::complex<double>, 1, n_mom, _nm> &mu,
                           SqMatArray<std::complex<double>, 1, n_mom, _nm>& residue);
    template <int n_mom>
    bool computeDefaultModel(const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& moms);
    void fixedPointRelation(const Eigen::Array<double, _n1, 1>& mats_freq, const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                            const SqMatArray<double, _n0, _n1, _nm>& Gwvar, const double m0trace, const double alpha,
                            const Eigen::Index s, SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm>& g) const;
    template <int n_mom>
    std::pair<bool, Eigen::Index> periodicPulaySolve(const Eigen::Array<double, _n1, 1>& mats_freq, const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                                                    const SqMatArray<double, _n0, _n1, _nm>& Gwvar, const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom,
                                                    const double alpha, const Eigen::Index s);
    template <typename Derived>
    double normInt(const Eigen::MatrixBase<Derived>& integrand) const {return m_intA.dot(integrand.cwiseAbs().colwise().sum());}
    template <typename Derived, typename OtherDerived>
    double dHMaxMag(const Eigen::MatrixBase<Derived>& Gwvar, const double m0trace, const double alpha, const Eigen::MatrixBase<OtherDerived>& dA) const;
    // Calculate chi^2
    double misfit(const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw, const SqMatArray<double, _n0, _n1, _nm>& Gwvar, const Eigen::Index s) const;
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
    const auto amaxtrial = std::any_cast<Eigen::Index>(parameters.at("alpha_max_trial"));
    const auto astopslope = std::any_cast<double>(parameters.at("alpha_stop_slope"));
    const auto astopstep = std::any_cast<double>(parameters.at("alpha_stop_step"));
    const auto verbose = std::any_cast<bool>(parameters.at("verbose"));
    const auto dAtol = std::any_cast<double>(parameters.at("alpha_spec_rel_err"));
    const auto rmin = std::any_cast<double>(parameters.at("alpha_step_min_ratio"));
    const auto rmax = std::any_cast<double>(parameters.at("alpha_step_max_ratio"));
    const auto sa = std::any_cast<double>(parameters.at("alpha_step_scale"));
    const auto acapacity = std::any_cast<Eigen::Index>(parameters.at("alpha_capacity"));
    const auto acacheall = std::any_cast<bool>(parameters.at("alpha_cache_all"));
    //const auto afitsize = std::any_cast<Eigen::Index>(parameters.at("alpha_curvature_fit_size"));
    const double eps = 1e-10;
    if (amaxfac < ainfofitfac) throw std::range_error("computeSpectra: alpha_max_fac should not be smaller than alpha_info_fit_fac");
    if (amaxtrial < 1) throw std::range_error("computeSpectra: num_alpha cannot be less than 1");
    
    double m0trace, varmin, logainfofit, loga, logchi2, logchi2_old, dloga, dloga_fac, dAr, dH, slope;
    Eigen::Index na, nrecord, trial, s;
    std::pair<bool, Eigen::Index> cvg;
    
    computeDefaultModel(mom);  // m_D, m_log_normD allocated and calculated in here
    if (Gw.procRank() == 0) {
        printData("default_model.txt", m_D);
        std::cout << "Output default_model.txt" << std::endl;
    }
    
    m_A.mpiComm(Gw.mpiComm());
    m_A.resize(Gw.dim0(), m_omega.size(), Gw.dimm());
    
    const auto Gwpart = Gw.mastDim0Part();
    const auto Gwvarpart = Gwvar.mastDim0Part();
    auto Apart = m_A.mastDim0Part();
    auto Dpart = m_D.mastDim0Part();
    std::vector<SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm> > As(acapacity);
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm> A_old(1, m_A.dim1(), m_A.dimm()), dA(1, m_A.dim1(), m_A.dimm());
    //Eigen::ArrayXd curvature, deriv, ainterval;
    bool converged = true;
    m_opt_alpha_id.resize(Gwpart.dim0());
    m_misfit_curve.resize(Gwpart.dim0());
    Apart() = Dpart();  // Initialize m_A
    std::ostringstream ostr;
    if (verbose && Gw.procRank() == 0) {
        ostr.copyfmt(std::cout);
        std::cout << std::scientific << std::setprecision(3);
    }
    for (Eigen::Index sl = 0; sl < Gwpart.dim0(); ++sl) {
        m_pulay_mix_param = std::any_cast<double>(parameters.at("Pulay_mixing_param"));
        s = sl + Gwpart.displ();
        m0trace = mom(s, 0).trace().real();
        // Initialize quantities
        na = 0;
        nrecord = 0;
        trial = 0;
        dloga = 0.0;  // Set zero for initial calculation
        slope = 10.0 * std::abs(astopslope);
        varmin = Gwvarpart.atDim0(sl).minCoeff();
        loga = std::log10(amaxfac / varmin);
        logainfofit = std::log10(ainfofitfac / varmin);
        cvg.first = true;
        m_opt_alpha_id(sl) = 0;
        logchi2 = std::log10(misfit(Gw, Gwvar, s));
        m_misfit_curve[sl].resize(acapacity, Eigen::NoChange);
        if (verbose && Gw.procRank() == 0) std::cout << "------ MQEM: decreasing alpha for spin " << s << " ------" << std::endl
            << "    stepID log10alpha log10chi^2   stepSize      slope #PulayIter #PulayFail" << std::endl;
        do {
            if (cvg.first) A_old() = Apart.atDim0(sl);
            cvg = periodicPulaySolve(mats_freq, Gw, Gwvar, mom, std::pow(10.0, loga - dloga), s);  // m_A updated in here
            if (!cvg.first) {  // Solve diverged
                if (na == 0) {  // Initial solve
                    converged = false;
                    std::cout << "MQEM computeSpectra: solving for initial alpha diverged for spin " << s << std::endl;
                    break;
                }
                Apart.atDim0(sl) = A_old();  // Restore initial guess
                dloga *= rmin;
                //parameters.at("Pulay_period") = std::any_cast<Eigen::Index>(parameters.at("Pulay_period")) + 1;
                m_pulay_mix_param *= rmin;
                ++trial;
                if (trial > amaxtrial) {
                    converged = false;
                    std::cout << "MQEM computeSpectra: maximum number of trials reached (diverged) for spin " << s << std::endl;
                    break;
                }
                else continue;
            }
            // Solve converged
            loga -= dloga;
            logchi2_old = logchi2;
            logchi2 = std::log10(misfit(Gw, Gwvar, s));
            // Initial slope set to zero to pretend that initial alpha is large enough for spectrum to be close enough to default model
            slope = na == 0 ? 0.0 : (logchi2_old - logchi2) / dloga;
            
            if (acacheall || loga <= logainfofit) {
                As[nrecord].resize(1, m_A.dim1(), m_A.dimm());
                As[nrecord]() = Apart.atDim0(sl);
                m_misfit_curve[sl](nrecord, 0) = loga;
                m_misfit_curve[sl](nrecord, 1) = logchi2;
                ++nrecord;
            }
            
            if (verbose && Gw.procRank() == 0) std::cout << std::setw(10) << na << " " << std::setw(10) << loga << " " << std::setw(10) << logchi2
                << " " << std::setw(10) << dloga << " " << std::setw(10) << slope << " " << std::setw(10) << cvg.second
                << " " << std::setw(10) << trial << std::endl;
            
            // Update step size
            if (na == 0) dloga = (loga - logainfofit) * ainitfrac;  // Really initialize step size here
            else {
                dA() = Apart.atDim0(sl) - A_old();
                //dAr = std::sqrt((Apart.atDim0(s) - A_old()).cwiseAbs2().sum() / A_old().cwiseAbs2().sum());
                dAr = (dA().array().abs() / (A_old().array().abs() + eps)).maxCoeff();
                //dAr = normInt(Apart.dim1RowVecsAtDim0(s) - A_old.dim1RowVecsAtDim0(0)) / normInt(A_old.dim1RowVecsAtDim0(0));
                dH = dHMaxMag(Gwvarpart.dim1RowVecsAtDim0(sl), m0trace, std::pow(10.0, loga + dloga), dA.dim1RowVecsAtDim0(0));
                //if (dA < dAtol) parameters.at("Pulay_period") = std::max(std::any_cast<Eigen::Index>(parameters.at("Pulay_period")) - 1, Eigen::Index(2));
                dloga_fac = std::min(std::max(sa * (dAtol - dH) / (dAr + dH), rmin), rmax);
                dloga *= dloga_fac;
                m_pulay_mix_param *= dloga_fac;
            }
            
            ++na;
            trial = 0;
        } while ((slope > astopslope || loga > logainfofit) && dloga > astopstep && nrecord < acapacity);
        if (verbose && Gw.procRank() == 0) {
            std::cout << "------ Stopped due to ";
            if (slope <= astopslope) std::cout << "small slope";
            else if (dloga <= astopstep) std::cout << "small step";
            else if (nrecord >= acapacity) std::cout << "full reservoir";
            else std::cout << "divergence";
            std::cout << " ------" << std::endl;
        }
        m_misfit_curve[sl].conservativeResize(nrecord, Eigen::NoChange);  // Get size right
        /*
        if (nrecord >= afitsize) {  // Calculate misfit curve curvature and find optimal alpha and spectrum
            //m_misfit_curve[s](0, 2) = std::nan("initial");
            //m_misfit_curve[s](1, 2) = std::nan("initial");
            //ainterval = m_misfit_curve[s](Eigen::seq(0, na - 2), 0) - m_misfit_curve[s](Eigen::seq(1, na - 1), 0);
            //deriv = (m_misfit_curve[s](Eigen::seq(0, na - 2), 1) - m_misfit_curve[s](Eigen::seq(1, na - 1), 1)) / ainterval;  // Forward first-order derivative
            //m_misfit_curve[s](Eigen::seq(2, na - 1), 2) = (deriv.head(na - 2) - deriv.tail(na - 2)) / ainterval.tail(na - 2);  // Forward second-order derivative
            //m_misfit_curve[s](Eigen::seq(2, na - 1), 2) /= (1.0 + deriv.tail(na - 2).square()).cube().sqrt();  // Signed curvature
            //m_misfit_curve[s](Eigen::seq(2, na - 1), 2).maxCoeff(&(m_opt_alpha_id(s)));
            //m_opt_alpha_id(s) += 2;
            ((m_misfit_curve[sl](Eigen::seq(0, Eigen::last - 1), 1) - m_misfit_curve[sl](Eigen::seq(1, Eigen::last), 1)) /
             (m_misfit_curve[sl](Eigen::seq(0, Eigen::last - 1), 0) - m_misfit_curve[sl](Eigen::seq(1, Eigen::last), 0))).maxCoeff(&max_slope_id);
            max_slope_id += afitsize / 2;
            if (max_slope_id >= nrecord) throw std::range_error("MQEM: Calculated misfit curve did not pass information-fitting regime");
            fitCurvature(m_misfit_curve[sl].template leftCols<2>(), m_misfit_curve[sl].col(2), afitsize);
            m_misfit_curve[sl](Eigen::seq(max_slope_id, Eigen::last), 2).template maxCoeff<Eigen::PropagateNumbers>(&(m_opt_alpha_id(sl)));
            m_opt_alpha_id(sl) += max_slope_id;
            Apart.atDim0(sl) = As[m_opt_alpha_id(sl)]();
        }
        else throw std::runtime_error("MQEM computeSpectra: cannot determine optimal alpha and spectrum because #points in misfit curve is less than local fit size");
         */
        //fitCubicSpline(m_misfit_curve[sl](Eigen::seq(Eigen::last, 0, Eigen::fix<-1>), Eigen::seqN(0, Eigen::fix<2>)),
        //               m_misfit_curve[sl](Eigen::seq(Eigen::last, 0, Eigen::fix<-1>), Eigen::seqN(2, Eigen::fix<2>)));  // Note reversed order
        fitFDFunc(m_misfit_curve[sl].template leftCols<2>(), m_misfit_curve[sl].template rightCols<2>());
        m_misfit_curve[sl].col(3).maxCoeff(&(m_opt_alpha_id(sl)));
        Apart.atDim0(sl) = As[m_opt_alpha_id(sl)]();
    }
    Apart.allGather();
    if (verbose && Gw.procRank() == 0) std::cout.copyfmt(ostr);
    return converged;  // Each process return its own convergence status
}

template <int _n0, int _n1, int _nm>
void MQEMContinuator<_n0, _n1, _nm>::computeRetardedFunc() {
    m_G_retarded.mpiComm(m_A.mpiComm());
    m_G_retarded.resize(m_A.dim0(), m_A.dim1(), m_A.dimm());
    auto GRpart = m_G_retarded.mastDim0Part();
    auto Apart = m_A.mastDim0Part();
    GRpart() = -1i * M_PI * Apart();
    for (Eigen::Index s = 0; s < Apart.dim0(); ++s) GRpart.dim1RowVecsAtDim0(s).noalias() += Apart.dim1RowVecsAtDim0(s) * m_Kp;  // Principle integral
    GRpart.allGather();
}

template <int _n0, int _n1, int _nm>
void MQEMContinuator<_n0, _n1, _nm>::computeRetardedFunc(const SqMatArray<std::complex<double>, _n0, 1, _nm> &static_part) {
    computeRetardedFunc();
    for (Eigen::Index i = 0; i < m_G_retarded.dim1(); ++i) m_G_retarded.atDim1(i) += static_part();
}

template <int _n0, int _n1, int _nm>
template <typename Derived>
Eigen::ArrayXd MQEMContinuator<_n0, _n1, _nm>::midRealFreqs(const Eigen::ArrayBase<Derived> &anchors_steps) {
    if (anchors_steps.size() < 3 || anchors_steps.size() % 2 == 0)
        throw std::range_error("MQEMContinuator::assembleMidRealFreqs: size of anchor_interval array is wrong");
    
    const Eigen::Index n_intervals = (anchors_steps.size() - 1) / 2;
    const Eigen::Index n_mid_anchors = n_intervals - 1;
    const Eigen::Index n_half_trans = 5;
    const Eigen::Index n_trans = 2 * n_half_trans;
    
    
    // Prepare steps in transition zone
    Eigen::ArrayXXd trans_dw(n_trans, n_mid_anchors);
    for (Eigen::Index j = 0; j < n_mid_anchors; ++j)
        for (Eigen::Index i = 0; i < n_trans; ++i)
            trans_dw(i, j) = 0.5 * (anchors_steps(2 * j + 1) + anchors_steps(2 * j + 3)
                                    + (anchors_steps(2 * j + 3) - anchors_steps(2 * j + 1))
                                    * std::tanh(2.0 * (static_cast<double>(i) - static_cast<double>(n_half_trans) + 0.5) / static_cast<double>(n_half_trans)));
    // Prepare steps in constant zone
    double const_interval;
    ArrayXindex n_const_dw(n_intervals);
    Eigen::ArrayXd const_dw(n_intervals);
    for (Eigen::Index i = 0; i < n_intervals; ++i) {
        const_interval = anchors_steps(2 * i + 2) - anchors_steps(2 * i);
        if (i > 0) const_interval -= trans_dw(Eigen::lastN(n_half_trans), i - 1).sum();
        if (i < n_intervals - 1) const_interval -= trans_dw(Eigen::seqN(0, n_half_trans), i).sum();
        if (const_interval < 0.0)
            throw std::runtime_error("MQEMContinuator::assembleMidRealFreqs: encountered negative interval for constant range; try to reduce step size");
        n_const_dw(i) = std::max<Eigen::Index>(static_cast<Eigen::Index>(const_interval / anchors_steps(2 * i + 1) + 0.5), 1);
        const_dw(i) = const_interval / n_const_dw(i);
    }
    
    // Calculate real frequency grid
    const Eigen::Index n_realfreqs = n_trans * n_mid_anchors + n_const_dw.sum() + 1;
    Eigen::Index k = 0;
    Eigen::ArrayXd realfreqs(n_realfreqs);
    realfreqs(0) = anchors_steps(0);
    for (Eigen::Index j = 0; j < n_mid_anchors; ++j) {
        for (Eigen::Index i = 0; i < n_const_dw(j); ++i) {
            realfreqs(k + 1) = realfreqs(k) + const_dw(j);
            ++k;
        }
        for (Eigen::Index i = 0; i < n_trans; ++i) {
            realfreqs(k + 1) = realfreqs(k) + trans_dw(i, j);
            ++k;
        }
    }
    for (Eigen::Index i = 0; i < n_const_dw(n_intervals - 1); ++i) {
        realfreqs(k + 1) = realfreqs(k) + const_dw(n_intervals - 1);
        ++k;
    }
    
    return realfreqs;
}

template <int _n0, int _n1, int _nm>
template <typename Derived>
void MQEMContinuator<_n0, _n1, _nm>::assembleKernelMatrix(const Eigen::Array<double, _n1, 1> &mats_freq, const Eigen::Index Nul,
                                                          const Eigen::ArrayBase<Derived>& real_freq_mid, const Eigen::Index Nur) {
    const Eigen::Index Nw = real_freq_mid.size();
    if (Nul < 1) throw std::range_error("computeKernelMatrix: Nul cannot be less than 1");
    if (Nw < 2) throw std::range_error("computeKernelMatrix: #points in middle part of real frequencies cannot be less than 2");
    if (Nur < 1) throw std::range_error("computeKernelMatrix: Nur cannot be less than 1");
    
    const auto eps = std::any_cast<double>(parameters.at("principle_int_eps"));
    const double domegal = real_freq_mid(1) - real_freq_mid(0);
    const double domegar = real_freq_mid(Nw - 1) - real_freq_mid(Nw - 2);
    const double omega0l = real_freq_mid(0) + Nul * domegal;
    // This is negative; point sequence dul, 2*dul, ..., Nul*dul corresponds omega from left to right
    const double dul = 1.0 / (Nul * (real_freq_mid(0) - domegal - omega0l));
    const double omega0r = real_freq_mid(Nw - 1) - Nur * domegar;
    // This is positive; point sequence dur, 2*dur, ..., Nur*dur corresponds omega from right to left
    const double dur = 1.0 / (Nur * (real_freq_mid(Nw - 1) + domegar - omega0r));
    
    const Eigen::Index n_iomega = mats_freq.size();
    const Eigen::Index Nspl = Nul + Nw - 1 + Nur;
    const Eigen::Index Ncoeff = 4 * Nspl;
    const Eigen::Index n_omega = Nspl + 1;
    // Matrix connecting Matsubara Green's function and cubic spline coefficients of spectral function
    Eigen::Matrix<std::complex<double>, _n1, Eigen::Dynamic> K(n_iomega, Ncoeff);  // Every element of K will be explicitly set, so no need of initialization
    Eigen::MatrixXd Kp(n_omega, Ncoeff);
    Eigen::VectorXd intA(Ncoeff), intwA(Ncoeff), intw2A(Ncoeff);  // Every element will be explicitly set, so no need of initialization
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(Ncoeff, Ncoeff), T = Eigen::MatrixXd::Zero(Ncoeff, n_omega);
    double uj, uj1 = 2.0 * dul, ujp, uj1p, omegajp, omegaj1p, domegaj;
    Eigen::Index ja = 0, jglobal;
    
    // First fill real frequencies; all the real frequencies will be needed later
    m_omega.resize(n_omega);
    for (Eigen::Index j = 0; j < Nul; ++j) m_omega(j) = 1.0 / ((j + 1) * dul) + omega0l;
    m_omega(Eigen::seqN(Nul, Nw)) = real_freq_mid;
    for (Eigen::Index j = 0; j < Nur; ++j) m_omega(j + Nul + Nw) = 1.0 / ((Nur - j) * dur) + omega0r;
    
    m_K_raw_Hc.resize(n_iomega, n_omega);
    for (Eigen::Index j = 0; j < n_omega; ++j)
        for (Eigen::Index n = 0; n < n_iomega; ++n)
            m_K_raw_Hc(n, j) = -1.0 / (1i * mats_freq(n) + m_omega(j));
    
    // For left side
    for (Eigen::Index j = 0; j < Nul; ++j) {  // The number of spline is Nul for the left side
        ja = 4 * j;
        uj = (j + 1) * dul;
        uj1 = (j + 2) * dul;
        //m_omega(j) = 1.0 / uj + omega0l;
        for (Eigen::Index n = 0; n < n_iomega; ++n) {
            K(n, ja) = a_side_coeff(uj, uj1, 1i * mats_freq(n), omega0l);
            K(n, ja + 1) = b_side_coeff(uj, uj1, 1i * mats_freq(n), omega0l);
            K(n, ja + 2) = c_side_coeff(uj, uj1, 1i * mats_freq(n), omega0l);
            K(n, ja + 3) = d_side_coeff(uj, uj1, 1i * mats_freq(n), omega0l);
        }
        for (Eigen::Index i = 0; i < n_omega; ++i) {
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
        intwA(ja) = uj - uj1 + omega0l * (uj * uj - uj1 * uj1) / 2.0;
        intwA(ja + 1) = std::log(uj / uj1) + omega0l * (uj - uj1);
        intwA(ja + 2) = 1.0 / uj1 - 1.0 / uj + omega0l * std::log(uj / uj1);
        intwA(ja + 3) = (1.0 / (uj1 * uj1) - 1.0 / (uj * uj)) / 2.0 + omega0l * (1.0 / uj1 - 1.0 / uj);
        intw2A(ja) = std::log(uj / uj1) + 2.0 * omega0l * (uj - uj1) + omega0l * omega0l * (uj * uj - uj1 * uj1) / 2.0;
        intw2A(ja + 1) = 1.0 / uj1 - 1.0 / uj + 2.0 * omega0l * std::log(uj / uj1) + omega0l * omega0l * (uj - uj1);
        intw2A(ja + 2) = (1.0 / (uj1 * uj1) - 1.0 / (uj * uj)) / 2.0 + 2.0 * omega0l * (1.0 / uj1 - 1.0 / uj) + omega0l * omega0l * std::log(uj / uj1);
        intw2A(ja + 3) = (1.0 / (uj1 * uj1 * uj1) - 1.0 / (uj * uj * uj)) / 3.0 + omega0l * (1.0 / (uj1 * uj1) - 1.0 / (uj * uj)) + omega0l * omega0l * (1.0 / uj1 - 1.0 / uj);
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
    for (Eigen::Index j = 0; j < Nw - 1; ++j) {  // The number of spline is Nw - 1 for the middle part
        jglobal = j + Nul;
        ja = 4 * jglobal;
        //omegaj = j * domega + omegal;
        //m_omega(jglobal) = omegaj;
        for (Eigen::Index n = 0; n < n_iomega; ++n) {
            K(n, ja) = a_mid_coeff(m_omega(jglobal), m_omega(jglobal + 1), 1i * mats_freq(n));
            K(n, ja + 1) = b_mid_coeff(m_omega(jglobal), m_omega(jglobal + 1), 1i * mats_freq(n));
            K(n, ja + 2) = c_mid_coeff(m_omega(jglobal), m_omega(jglobal + 1), 1i * mats_freq(n));
            K(n, ja + 3) = d_mid_coeff(m_omega(jglobal), m_omega(jglobal + 1), 1i * mats_freq(n));
        }
        for (Eigen::Index i = 0; i < n_omega; ++i) {
            if (i == jglobal + 1) {  // Spline immediately left to point i
                omegajp = m_omega(jglobal);
                omegaj1p = m_omega(jglobal + 1) - eps;
            }
            else if (i == jglobal) {  // Spline immediately right to point i
                omegajp = m_omega(jglobal) + eps;
                omegaj1p = m_omega(jglobal + 1);
            }
            else {
                omegajp = m_omega(jglobal);
                omegaj1p = m_omega(jglobal + 1);
            }
            Kp(i, ja) = a_mid_coeff(omegajp, omegaj1p, m_omega(i));
            Kp(i, ja + 1) = b_mid_coeff(omegajp, omegaj1p, m_omega(i));
            Kp(i, ja + 2) = c_mid_coeff(omegajp, omegaj1p, m_omega(i));
            Kp(i, ja + 3) = d_mid_coeff(omegajp, omegaj1p, m_omega(i));
        }
        domegaj = m_omega(jglobal + 1) - m_omega(jglobal);
        intA(ja) = domegaj * domegaj * domegaj * domegaj / 4.0; intA(ja + 1) = domegaj * domegaj * domegaj / 3.0; intA(ja + 2) = domegaj * domegaj / 2.0; intA(ja + 3) = domegaj;
        intwA(ja) = domegaj * domegaj * domegaj * domegaj * (domegaj / 5.0 + m_omega(jglobal) / 4.0);
        intwA(ja + 1) = domegaj * domegaj * domegaj * (domegaj / 4.0 + m_omega(jglobal) / 3.0);
        intwA(ja + 2) = domegaj * domegaj * (domegaj / 3.0 + m_omega(jglobal) / 2.0);
        intwA(ja + 3) = domegaj * (m_omega(jglobal + 1) + m_omega(jglobal)) / 2.0;
        intw2A(ja) = domegaj * domegaj * domegaj * domegaj * (domegaj * domegaj / 6.0 + 2.0 * domegaj * m_omega(jglobal) / 5.0 + m_omega(jglobal) * m_omega(jglobal) / 4.0);
        intw2A(ja + 1) = domegaj * domegaj * domegaj * (domegaj * domegaj / 5.0 + 2.0 * domegaj * m_omega(jglobal) / 4.0 + m_omega(jglobal) * m_omega(jglobal) / 3.0);
        intw2A(ja + 2) = domegaj * domegaj * (domegaj * domegaj / 4.0 + 2.0 * domegaj * m_omega(jglobal) / 3.0 + m_omega(jglobal) * m_omega(jglobal) / 2.0);
        intw2A(ja + 3) = domegaj * (m_omega(jglobal + 1) * m_omega(jglobal + 1) + m_omega(jglobal + 1) * m_omega(jglobal) + m_omega(jglobal) * m_omega(jglobal)) / 3.0;
        B(ja, ja + 3) = 1.0;
        T(ja, jglobal) = 1.0;
        B(ja + 1, ja) = domegaj * domegaj * domegaj; B(ja + 1, ja + 1) = domegaj * domegaj; B(ja + 1, ja + 2) = domegaj; B(ja + 1, ja + 3) = 1.0;
        T(ja + 1, jglobal + 1) = 1.0;
        if (j < Nw - 2) {
            B(ja + 2, ja) = 3.0 * domegaj * domegaj; B(ja + 2, ja + 1) = 2.0 * domegaj; B(ja + 2, ja + 2) = 1.0;
            B(ja + 2, ja + 6) = -1.0;
            B(ja + 3, ja) = 6.0 * domegaj; B(ja + 3, ja + 1) = 2.0;
            B(ja + 3, ja + 5) = -2.0;
        }
    }
    //m_omega(jglobal + 1) = omegar;
    
    const double ur = (Nur + 1) * dur;
    // For derivatives at omegar; ja just has the proper value now
    B(ja + 2, ja) = 3.0 * domegar * domegar; B(ja + 2, ja + 1) = 2.0 * domegar; B(ja + 2, ja + 2) = 1.0;
    // Arising from the first spline in the right side
    B(ja + 2, ja + 4) = ur * ur * 3.0 * ur * ur; B(ja + 2, ja + 5) = ur * ur * 2.0 * ur; B(ja + 2, ja + 6) = ur * ur;
    B(ja + 3, ja) = 6.0 * domegar; B(ja + 3, ja + 1) = 2.0;
    // Arising from the first spline in the right side
    B(ja + 3, ja + 4) = -12.0 * ur * ur * ur * ur * ur; B(ja + 3, ja + 5) = -6.0 * ur * ur * ur * ur; B(ja + 3, ja + 6) = -2.0 * ur * ur * ur;
    
    // For right side
    for (Eigen::Index j = 0; j < Nur; ++j) {
        jglobal = j + Nul + Nw - 1;
        ja = 4 * jglobal;
        uj = (Nur - j + 1) * dur;
        uj1 = (Nur - j) * dur;
        //m_omega(jglobal + 1) = 1.0 / uj1 + omega0r;
        for (Eigen::Index n = 0; n < n_iomega; ++n) {
            K(n, ja) = a_side_coeff(uj, uj1, 1i * mats_freq(n), omega0r);
            K(n, ja + 1) = b_side_coeff(uj, uj1, 1i * mats_freq(n), omega0r);
            K(n, ja + 2) = c_side_coeff(uj, uj1, 1i * mats_freq(n), omega0r);
            K(n, ja + 3) = d_side_coeff(uj, uj1, 1i * mats_freq(n), omega0r);
        }
        for (Eigen::Index i = 0; i < n_omega; ++i) {
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
        intwA(ja) = uj - uj1 + omega0r * (uj * uj - uj1 * uj1) / 2.0;
        intwA(ja + 1) = std::log(uj / uj1) + omega0r * (uj - uj1);
        intwA(ja + 2) = 1.0 / uj1 - 1.0 / uj + omega0r * std::log(uj / uj1);
        intwA(ja + 3) = (1.0 / (uj1 * uj1) - 1.0 / (uj * uj)) / 2.0 + omega0r * (1.0 / uj1 - 1.0 / uj);
        intw2A(ja) = std::log(uj / uj1) + 2.0 * omega0r * (uj - uj1) + omega0r * omega0r * (uj * uj - uj1 * uj1) / 2.0;
        intw2A(ja + 1) = 1.0 / uj1 - 1.0 / uj + 2.0 * omega0r * std::log(uj / uj1) + omega0r * omega0r * (uj - uj1);
        intw2A(ja + 2) = (1.0 / (uj1 * uj1) - 1.0 / (uj * uj)) / 2.0 + 2.0 * omega0r * (1.0 / uj1 - 1.0 / uj) + omega0r * omega0r * std::log(uj / uj1);
        intw2A(ja + 3) = (1.0 / (uj1 * uj1 * uj1) - 1.0 / (uj * uj * uj)) / 3.0 + omega0r * (1.0 / (uj1 * uj1) - 1.0 / (uj * uj)) + omega0r * omega0r * (1.0 / uj1 - 1.0 / uj);
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
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomp(B);
    //Eigen::FullPivLU<Eigen::MatrixXd> decomp(B);
    const auto thr = std::any_cast<double>(parameters.at("matrix_invertible_threshold"));
    if (thr >= 0.0) decomp.setThreshold(thr);
    //if (decomp.isInvertible()) B = decomp.inverse();
    //else throw std::runtime_error("MQEMContinuator::assembleKernelMatrix: matrix B is not invertible");
    //const Eigen::MatrixXd tmp = B * T;  // Now B is already its inverse matrix
    const Eigen::MatrixXd tmp = decomp.solve(T);
    //std::cout << "Linear solving error: " << (T - B * tmp).cwiseAbs().maxCoeff() << std::endl;
    //assert(T.isApprox(B * tmp));
    m_K.resize(n_omega, n_iomega);  // Note dimension order here
    m_Kp.resize(n_omega, n_omega);
    m_K.noalias() = tmp.transpose() * K.transpose();  // Note transpose here
    m_Kp.noalias() = tmp.transpose() * Kp.transpose();
    //Eigen::Index nanrow, nancol;
    //std::cout << Kp.array().isNaN().maxCoeff(&nanrow, &nancol) << std::endl;
    //std::cout << nanrow << ", " << nancol << std::endl;
    m_intA.resize(n_omega);
    m_intA.noalias() = tmp.transpose() * intA;
    m_intwA.resize(n_omega);
    m_intwA.noalias() = tmp.transpose() * intwA;
    m_intw2A.resize(n_omega);
    m_intw2A.noalias() = tmp.transpose() * intw2A;
}

// m_D and m_log_normD updated in here
template <int _n0, int _n1, int _nm>
template <int n_mom>
void MQEMContinuator<_n0, _n1, _nm>::momentConstraints(const SqMatArray<std::complex<double>, _n0, n_mom, _nm> &moms,
                                                       const Eigen::Index s, const SqMatArray<std::complex<double>, 1, n_mom, _nm> &mu,
                                                       SqMatArray<std::complex<double>, 1, n_mom, _nm> &residue) {
    residue.resize(1, moms.dim1(), moms.dimm());
    Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>, _nm, _nm> > ces(moms.dimm());
    for (Eigen::Index n = 0; n < m_D.dim1(); ++n) {
        m_log_normD(s, n) = mu[0] + mu[1] * m_omega(n) - mu[2] * m_omega(n) * m_omega(n);  // Not necessarilly Hermitian during iteration
        ces.compute(m_log_normD(s, n));
        m_D(s, n).noalias() = ces.eigenvectors() * ces.eigenvalues().array().exp().matrix().asDiagonal() * ces.eigenvectors().inverse();
    }
    residue[0] = (m_D.dim1RowVecsAtDim0(s) * m_intA).reshaped(moms.dimm(), moms.dimm()) - moms(s, 0);
    residue[1] = (m_D.dim1RowVecsAtDim0(s) * m_intwA).reshaped(moms.dimm(), moms.dimm()) - moms(s, 1);
    residue[2] = (m_D.dim1RowVecsAtDim0(s) * m_intw2A).reshaped(moms.dimm(), moms.dimm()) - moms(s, 2);
}

template <int _n0, int _n1, int _nm>
template <int n_mom>
bool MQEMContinuator<_n0, _n1, _nm>::computeDefaultModel(const SqMatArray<std::complex<double>, _n0, n_mom, _nm> &moms) {
    const auto simplemodel = std::any_cast<bool>(parameters.at("simple_default_model"));
    static constexpr int nm3 = _nm == Eigen::Dynamic ? Eigen::Dynamic : _nm * 3;
    if (moms.dim1() < 3) throw std::range_error("MQEMContinuator::computeDefaultModel: number of provided moments less than 3");
    const auto sigma = std::any_cast<double>(parameters.at("Gaussian_sigma"));
    const auto shift = std::any_cast<double>(parameters.at("Gaussian_shift"));
    const auto momspart = moms.mastDim0Part();
    double fac, logm0trace;
    m_D.mpiComm(moms.mpiComm());
    m_D.resize(moms.dim0(), m_omega.size(), moms.dimm());
    m_log_normD.mpiComm(moms.mpiComm());
    m_log_normD.resize(moms.dim0(), m_omega.size(), moms.dimm());
    auto Dpart = m_D.mastDim0Part();
    auto logDpart = m_log_normD.mastDim0Part();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<std::complex<double>, _nm, _nm> > es(moms.dimm());
    
    if (simplemodel) {
        for (Eigen::Index sl = 0; sl < Dpart.dim0(); ++sl) {
            logm0trace = momspart(sl, 0).trace().real();  // Notice this is not log
            es.compute(momspart(sl, 0));
            for (Eigen::Index n = 0; n < m_D.dim1(); ++n) {
                fac = std::exp(-(m_omega(n) - shift) * (m_omega(n) - shift) / (2.0 * sigma * sigma)) / sigma * M_SQRT1_2 * 0.5 * M_2_SQRTPI;
                Dpart(sl, n) = momspart(sl, 0) * fac;
                logDpart(sl, n).noalias() = es.eigenvectors() * (es.eigenvalues().array() * fac / logm0trace).log().matrix().asDiagonal() * es.eigenvectors().adjoint();
            }
        }
        Dpart.allGather();
        logDpart.allGather();
        return true;
    }
    
    for (Eigen::Index sl = 0; sl < Dpart.dim0(); ++sl) {
        es.compute(momspart(sl, 2), Eigen::EigenvaluesOnly);
        if ((es.eigenvalues().array() < 0.0).any())
            throw std::runtime_error("MQEMContinuator::computeDefaultModel: third moment of self-energy is not positive semi-definite; try to lower the tail's starting index for fitting the moments");
    }
    
    const auto max_iter = std::any_cast<Eigen::Index>(parameters.at("secant_max_iter"));
    const auto tol = std::any_cast<double>(parameters.at("secant_tol"));
    const auto damp = std::any_cast<double>(parameters.at("secant_damp"));
    const auto verbose = std::any_cast<bool>(parameters.at("verbose"));
    //const Eigen::Index max_inc_residue = 100;
    double err, err_old;
    SqMatArray<std::complex<double>, 1, 3, _nm> mu(1, 3, moms.dimm()), S(1, 3, moms.dimm()), residue(1, 3, moms.dimm()), residue_old(1, 3, moms.dimm());
    Eigen::Matrix<std::complex<double>, _nm, nm3> Y(moms.dimm(), moms.dimm() * 3);
    Eigen::Matrix<std::complex<double>, nm3, nm3> B(moms.dimm() * 3, moms.dimm() * 3);
    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<std::complex<double>, _nm, nm3> > decomp(moms.dimm(), moms.dimm() * 3);
    //const double sigma0 = 1.0;
    Eigen::Index s, iter;
    int stoptype;
    bool converged = true;
    Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>, _nm, _nm> > ces(moms.dimm());
    
    std::ostringstream ostr;
    if (verbose && m_D.procRank() == 0) {
        ostr.copyfmt(std::cout);
        std::cout << std::scientific << std::setprecision(3);
    }
    for (Eigen::Index sl = 0; sl < Dpart.dim0(); ++sl) {
        s = sl + Dpart.displ();
        if (verbose && m_D.procRank() == 0) {
            std::cout << "------ MQEM: computing default model for spin " << s << " ------" << std::endl;
            std::cout << "iter    residue" << std::endl;
        }
        // Initialize
        iter = 0;
        //n_inc_residue = 0;
        //mu_old[0] = (moms(s, 0).diagonal().array() / sigma0 * M_SQRT1_2 * 0.5 * M_2_SQRTPI).log().matrix().asDiagonal();
        //mu_old[1].setZero();
        //mu_old[2] = Eigen::Matrix<double, _nm, _nm>::Identity(moms.dimm(), moms.dimm()) / (2.0 * sigma0 * sigma0);
        //mu[0] = (moms(s, 0).diagonal().array() / sigma1 * M_SQRT1_2 * 0.5 * M_2_SQRTPI).log().matrix().asDiagonal();
        //mu[1] = Eigen::Matrix<double, _nm, _nm>::Ones(moms.dimm(), moms.dimm()) * 0.5;
        //mu[0].setRandom();
        //mu[1].setRandom();
        //mu[2] = Eigen::Matrix<double, _nm, _nm>::Identity(moms.dimm(), moms.dimm()) / (2.0 * sigma1 * sigma1);
        mu[2] = 0.5 * momspart(sl, 0) * momspart(sl, 0) * (momspart(sl, 2) * momspart(sl, 0) - momspart(sl, 1) * momspart(sl, 1)).inverse();
        mu[1] = 2.0 * mu[2] * momspart(sl, 1) * momspart(sl, 0).inverse();
        ces.compute(mu[2]);
        mu[0].noalias() = momspart(sl, 0) * ces.eigenvectors() * (ces.eigenvalues() / M_PI).cwiseSqrt().asDiagonal() * ces.eigenvectors().inverse();
        ces.compute(mu[0]);
        mu[0].noalias() = ces.eigenvectors() * ces.eigenvalues().array().log().matrix().asDiagonal() * ces.eigenvectors().inverse() - 0.25 * mu[1] * mu[1] * mu[2].inverse();
        //mu_old[2] = mu[2];
        //mu_old[1] = mu[1];
        //ces.compute(mu_old[0]);
        //mu_old[0].noalias() = tmp * ces.eigenvectors() * ces.eigenvalues().array().exp().matrix().asDiagonal() * ces.eigenvectors().inverse();
        //ces.compute(mu_old[0]);
        //mu_old[0].noalias() = ces.eigenvectors() * ces.eigenvalues().array().log().matrix().asDiagonal() * ces.eigenvectors().inverse();
        S() = mu().cwiseProduct(Eigen::Matrix<std::complex<double>, _nm, nm3>::Random(moms.dimm(), moms.dimm() * 3) * 0.1
                                + Eigen::Matrix<double, _nm, nm3>::Ones(moms.dimm(), moms.dimm() * 3));
        momentConstraints(moms, s, S, residue_old);
        err_old = residue_old().template lpNorm<Eigen::Infinity>();
        S() = mu() - S();
        
        while (true) {
            //momentConstraints(moms, s, mu_old, residue_old);
            momentConstraints(moms, s, mu, residue);
            
            //err_old = residue_old().template lpNorm<Eigen::Infinity>();
            err = residue().template lpNorm<Eigen::Infinity>();
            
            if (verbose && m_D.procRank() == 0) std::cout << std::setw(4) << iter << " " << std::setw(10) << err << std::endl;  // For testing
            if (!residue().array().isFinite().all()) {
                stoptype = 3;
                converged = false;
                break;
            }
            else if (err < tol) {
                stoptype = 0;
                break;
            }
            else if (err > err_old) {
                //++n_inc_residue;
                //mu() = mu_old() + damp * (S + 0.1 * Eigen::Matrix<std::complex<double>, _nm, nm3>::Random(moms.dimm(), moms.dimm() * 3));
                //if (n_inc_residue > max_inc_residue) {
                stoptype = 1;
                converged = false;
                break;
                //}
                //else continue;
            }
            else if (iter == max_iter) {
                stoptype = 2;
                converged = false;
                break;
            }
            
            // Secant method generalized for nonlinear matrix equations; update mu as a whole by approximating Jacobian with least square solution
            Y = residue() - residue_old();
            decomp.compute(Y);
            B = decomp.solve(S());
            S().noalias() = -residue() * B;
            //mu_old() = mu();
            mu() += damp * S();
            residue_old() = residue();
            err_old = err;
            
            //n_inc_residue = 0;
            ++iter;
        }
        if (stoptype == 1) {  // Use solution with smallest residue
            mu() -= damp * S();
            momentConstraints(moms, s, mu, residue);
        }
        if (verbose && m_D.procRank() == 0) {  // For testing
            std::cout << "------ Stopped due to ";
            if (stoptype == 0) std::cout << "convergence";
            else if (stoptype == 1) std::cout << "increasing residue";
            else if (stoptype == 2) std::cout << "full iteration";
            else if (stoptype == 3) std::cout << "divergence";
            std::cout << " ------" << std::endl;
            std::cout << "mu = " << std::endl;
            std::cout << mu << std::endl;  // mu is left as is, not made Hermitian
            std::cout << "residue = " << std::endl;
            std::cout << residue << std::endl;
        }
        
        logm0trace = std::log(momspart(sl, 0).trace().real());
        for (Eigen::Index n = 0; n < m_D.dim1(); ++n) {
            logDpart(sl, n) = (logDpart(sl, n) + logDpart(sl, n).adjoint().eval()) / 2.0;  // Make calculated default model Hermitian anyway
            es.compute(logDpart(sl, n));
            Dpart(sl, n).noalias() = es.eigenvectors() * es.eigenvalues().array().exp().matrix().asDiagonal() * es.eigenvectors().adjoint();
            logDpart(sl, n) -= logm0trace * Eigen::Matrix<double, _nm, _nm>::Identity(moms.dimm(), moms.dimm());  // Renormalize logD
        }
        
        // Check positive definiteness of mu[2], if not, degrade to the simple Gaussian
        es.compute((mu[2] + mu[2].adjoint()) / 2.0);
        if ((es.eigenvalues().array() <= 0.0).any()) {
            double fac;
            logm0trace = momspart(sl, 0).trace().real();  // Notice this is not log
            es.compute(momspart(sl, 0));
            for (Eigen::Index n = 0; n < m_D.dim1(); ++n) {
                fac = std::exp(-m_omega(n) * m_omega(n) / (2.0 * sigma * sigma)) / sigma * M_SQRT1_2 * 0.5 * M_2_SQRTPI;
                Dpart(sl, n) = momspart(sl, 0) * fac;
                logDpart(sl, n).noalias() = es.eigenvectors() * (es.eigenvalues().array() * fac / logm0trace).log().matrix().asDiagonal() * es.eigenvectors().adjoint();
            }
            std::cout << "Calculated mu[2] for spin " << s << " was not positive definite; degraded to simple Gaussian with standard deviation " << sigma << std::endl;
        }
    }
    Dpart.allGather();
    logDpart.allGather();
    if (verbose && m_D.procRank() == 0) std::cout.copyfmt(ostr);
    return converged;  // Each process return its own convergence status
}

// Renormalize quantities by m0trace in situ
template <int _n0, int _n1, int _nm>
void MQEMContinuator<_n0, _n1, _nm>::fixedPointRelation(const Eigen::Array<double, _n1, 1>& mats_freq,
                                                        const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                                                        const SqMatArray<double, _n0, _n1, _nm>& Gwvar,
                                                        double m0trace, const double alpha,
                                                        const Eigen::Index s,
                                                        SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm>& g) const {
    const Eigen::Index nm = Gw.dimm();
    const Eigen::Index n_omega = m_A.dim1();
    //Eigen::Matrix<double, _nm, Eigen::Dynamic> eigvals(nm, n_omega);
    Eigen::Matrix<std::complex<double>, nm2, _n1> chi = Gw.dim1RowVecsAtDim0(s);
    
    chi.noalias() -= m_A.dim1RowVecsAtDim0(s) * m_K;
    chi.array() *= (-m0trace / (2.0 * alpha)) / Gwvar.dim1RowVecsAtDim0(s).array();
    
    g.resize(1, n_omega, nm);
    g.dim1RowVecsAtDim0(0).noalias() = chi * m_K_raw_Hc;
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<std::complex<double>, _nm, _nm> > es(nm);
    Eigen::Matrix<std::complex<double>, _nm, _nm> tmp(nm, nm);
    double Z = 0.0;
    //Eigen::Array<double, _nm, 1> H1(nm);
    //double testmin = 0.0, testmax = 0.0;
    for (Eigen::Index i = 0; i < n_omega; ++i) {
        //for (Eigen::Index n = 0; n < n_iomega; ++n) g[i].noalias() += (m0trace * chi[n].array() / (2.0 * (1i * mats_freq(n) + m_omega(i)) * Gwvar(s, n).array() * alpha)).matrix();
        g[i] += g[i].adjoint().eval() - m_log_normD(s, i);  // Used eval() to explicitly evaluate to temporary so no aliasing issue
        es.compute(g[i]);
        //eigvals.col(i) = -es.eigenvalues();
        //g[i] = es.eigenvectors();
        tmp.noalias() = es.eigenvectors() * (-es.eigenvalues().array()).exp().matrix().asDiagonal();
        g[i].noalias() = tmp * es.eigenvectors().adjoint();
        Z += m_intA(i) * g[i].trace().real();  // Trace must be real
    }
    
    //const auto limexp = std::any_cast<double>(parameters.at("Pulay_exp_limit"));
    //double maxexp, offset;
    //maxexp = eigvals.maxCoeff();
    //offset = maxexp > limexp ? maxexp - limexp : 0.0;
    //for (Eigen::Index i = 0; i < n_omega; ++i) {
    //    g[i] = g[i] * (eigvals.col(i).array() - offset).exp().matrix().asDiagonal() * g[i].adjoint();
    //    gtr(i) = g[i].trace().real();  // Trace must be real
    //}
    g() *= m0trace / Z;
}

// Not initializing m_A, just taking whatever in m_A as the initial guess, which is advantageous for concecutive calculations as varying m_alpha smoothly
template <int _n0, int _n1, int _nm>
template <int n_mom>
std::pair<bool, Eigen::Index> MQEMContinuator<_n0, _n1, _nm>::periodicPulaySolve(const Eigen::Array<double, _n1, 1>& mats_freq,
                                                                                const SqMatArray<std::complex<double>, _n0, _n1, _nm>& Gw,
                                                                                const SqMatArray<double, _n0, _n1, _nm>& Gwvar,
                                                                                const SqMatArray<std::complex<double>, _n0, n_mom, _nm>& mom,
                                                                                const double alpha,
                                                                                const Eigen::Index s) {
    //const auto mix_param = std::any_cast<double>(parameters.at("Pulay_mixing_param"));
    const auto hist_size = std::any_cast<Eigen::Index>(parameters.at("Pulay_history_size"));
    const auto period = std::any_cast<Eigen::Index>(parameters.at("Pulay_period"));
    const auto tol = std::any_cast<double>(parameters.at("Pulay_tolerance"));
    const auto max_iter = std::any_cast<Eigen::Index>(parameters.at("Pulay_max_iteration"));
    if (hist_size == 0) throw std::range_error("periodicPulaySolve: history size cannot be zero");
    if (period == 0) throw std::range_error("periodicPulaySolve: period cannot be zero");
    
    const Eigen::Index n_omega = m_A.dim1();
    const Eigen::Index nm = m_A.dimm();
    const Eigen::Index N = n_omega * nm * nm;
    // No need to initialize; use these only after they are fully filled
    Eigen::MatrixXcd R(N, hist_size), F(N, hist_size), RF(N, hist_size);
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, _nm> A_old(1, n_omega, nm), f(1, n_omega, nm), f_old(1, n_omega, nm);
    Eigen::Index iter = 0, ih = 0;
    double err;
    const double m0trace = mom(s, 0).trace().real();  // Trace must be a real number;
    //Eigen::PermutationMatrix<Eigen::Dynamic> perm(hist_size);
    //Eigen::MatrixXcd J(hist_size, N);  // FTF_inv(hist_size, hist_size);
    Eigen::VectorXcd v(hist_size);
    //Eigen::LLT<Eigen::MatrixXcd> llt(hist_size);
    //Eigen::PartialPivLU<Eigen::MatrixXcd> decomp(hist_size);
    //Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> decomp(hist_size, hist_size);
    //Eigen::FullPivLU<Eigen::MatrixXcd> decomp(hist_size, hist_size);
    //Eigen::FullPivHouseholderQR<Eigen::MatrixXcd> decomp(hist_size, hist_size);
    //Eigen::JacobiSVD<Eigen::MatrixXcd, Eigen::NoQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeThinV> decomp(hist_size, hist_size);
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXcd> decomp(N, hist_size);
    
    while (true) {
        // Calculate some current variables for computing new ones
        fixedPointRelation(mats_freq, Gw, Gwvar, m0trace, alpha, s, f);   // Read m_A in here
        f() -= m_A.atDim0(s);
        //err = std::sqrt(f().cwiseAbs2().sum() / n_omega);
        err = normInt(f.dim1RowVecsAtDim0(0));
        if (!std::isfinite(err)) return std::make_pair(false, iter);
        else if (err < tol) return std::make_pair(true, iter);
        else if (iter == max_iter) return std::make_pair(false, iter);
        //std::cout << "error = " << err << std::endl;
        // Record current variables into history caches
        if (iter > 0 && period - iter % period <= hist_size) {
            R.col(ih) = (m_A.atDim0(s) - A_old()).reshaped();  // RHS flattened to a vector
            F.col(ih) = (f() - f_old()).reshaped();
            ih = (ih + 1) % hist_size;
        }
        // Record current variables
        A_old() = m_A.atDim0(s);
        f_old() = f();
        // Calculate new variables
        m_A.atDim0(s) += m_pulay_mix_param * f();  // Linear mixing
        if (iter >= hist_size && (iter + 1) % period == 0) {  // Pulay mixing, do this only after history caches are full
            // Prepare permutation matrix to properly order the history
            // for (Eigen::Index ip = 0; ip < hist_size; ++ip) perm.indices()[ip] = (ip + 1 + ih) % hist_size;
            // The formula is invariant under simultaneous column permutation of R and F, so no need to reorder the columns
            //FTF_inv.noalias() = F.adjoint() * F;
            //llt.compute(FTF_inv);
            //FTF_inv = llt.solve(Eigen::MatrixXcd::Identity(hist_size, hist_size));  // Inverse of FTF
            //m_A.atDim0(s).noalias() += mix_param * f() - ((R + mix_param * F) * FTF_inv * F.adjoint() * f().reshaped()).reshaped(nm, nm * n_omega);
            //FTF_inv.noalias() = F.transpose() * F;
            //decomp.compute(FTF_inv);
            //FTF_inv = decomp.solve(Eigen::MatrixXcd::Identity(hist_size, hist_size));  // Inverse of FTF
            //if (decomp.isInvertible()) FTF_inv = decomp.inverse();
            //else throw std::runtime_error("Periodic Pulay solver: F^T * F not invertible");
            //m_A.atDim0(s).noalias() += mix_param * f() - ((R + mix_param * F) * FTF_inv * F.transpose() * f().reshaped()).reshaped(nm, nm * n_omega);
            decomp.compute(F);
            //J = decomp.solve(Eigen::MatrixXcd::Identity(N, N));
            //v.noalias() = J * f().reshaped();
            v = decomp.solve(f().reshaped());
            RF = R + m_pulay_mix_param * F;
            m_A.atDim0(s).reshaped().noalias() -= RF * v;
        }
        ++iter;
    }
}

template <int _n0, int _n1, int _nm>
template <typename Derived, typename OtherDerived>
double MQEMContinuator<_n0, _n1, _nm>::dHMaxMag(const Eigen::MatrixBase<Derived> &Gwvar, const double m0trace, const double alpha,
                                                const Eigen::MatrixBase<OtherDerived> &dA) const {
    Eigen::Matrix<typename OtherDerived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> AK = dA * m_K;
    AK.array() *= (m0trace / (2.0 * alpha)) / Gwvar.array();
    SqMatArray<typename OtherDerived::Scalar, 1, OtherDerived::ColsAtCompileTime, _nm> dH(1, dA.cols(), static_cast<Eigen::Index>(std::sqrt(dA.rows()) + 0.5));
    dH.dim1RowVecsAtDim0(0).noalias() = AK * m_K_raw_Hc;
    for (Eigen::Index i = 0; i < dH.dim1(); ++i) dH[i] += dH[i].adjoint().eval();
    return dH().template lpNorm<Eigen::Infinity>();
}

template <int _n0, int _n1, int _nm>
double MQEMContinuator<_n0, _n1, _nm>::misfit(const SqMatArray<std::complex<double>, _n0, _n1, _nm> &Gw, const SqMatArray<double, _n0, _n1, _nm> &Gwvar,
                                              const Eigen::Index s) const {
    Eigen::Matrix<std::complex<double>, nm2, _n1> dG = Gw.dim1RowVecsAtDim0(s);
    dG.noalias() -= m_A.dim1RowVecsAtDim0(s) * m_K;
    return (dG.array().abs2() / Gwvar.dim1RowVecsAtDim0(s).array()).sum();
}

template <int _n0, int _n1, int _nm>
template <typename Derived, typename OtherDerived>
void MQEMContinuator<_n0, _n1, _nm>::fitCurvature(const Eigen::DenseBase<Derived> &curve, const Eigen::DenseBase<OtherDerived> &curvature,
                                                  const Eigen::Index n_fitpts) {
    if (curve.rows() < n_fitpts) throw std::range_error("MQEMContinuator::fitCurvature: #points of input curve cannot be less than #local points to fit");
    if (n_fitpts < 3) throw std::range_error("MQEMContinuator::fitCurvature: #local points to fit cannot be less than 3");
    
    typedef typename Derived::Scalar Scalar;
    //typedef typename Eigen::internal::plain_col_type<Derived>::type VectorType;
    
    // We choose circle equation A * (x^2 + y^2) + B * x + C * y + 1 = 0, so it can also represent a straight line.
    // It has 3 parameters; 3 for #row for contiguous memory for local param matrix
    Eigen::Matrix<Scalar, 3, Derived::RowsAtCompileTime> param_mat(3, curve.rows());
    Eigen::Vector<Scalar, Derived::RowsAtCompileTime> param_vec(curve.rows());
    
    // Assemble the full param matrix and vector
    for (Eigen::Index i = 0; i < curve.rows(); ++i) {
        //param_mat(0, i) = curve(i, 0);
        //param_mat(1, i) = curve(i, 1);
        //param_mat(2, i) = 1.0;
        //param_vec(i) = -(curve(i, 0) * curve(i, 0) + curve(i, 1) * curve(i, 1));
        param_mat(0, i) = curve(i, 0) * curve(i, 0) + curve(i, 1) * curve(i, 1);
        param_mat(1, i) = curve(i, 0);
        param_mat(2, i) = curve(i, 1);
        param_vec(i) = -1.0;
    }
    
    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<Scalar, 3, Eigen::Dynamic> > decomp(3, n_fitpts);
    Eigen::Vector<Scalar, 3> circ_coeffs;
    Scalar dx, dy, centerx, centery;
    const Eigen::Index ifitbegin = n_fitpts / 2;
    const Eigen::Index ifitend = curve.rows() - (n_fitpts + 1) / 2;
    
    Eigen::DenseBase<OtherDerived>& curvature_ = const_cast<Eigen::DenseBase<OtherDerived>&>(curvature);  // Make curvature writable by casting away its const-ness
    curvature_.derived().resize(curve.rows());  // Resize the derived object
    
    // Calculate curvature by least square fitting (solving)
    for (Eigen::Index i = 0; i < curve.rows(); ++i) {
        if (i < ifitbegin) curvature_(i) = std::nan("initial");
        else if (i > ifitend) curvature_(i) = std::nan("last");
        else {
            dx = curve(i + 1, 0) - curve(i, 0);
            dy = (curve(i + 1, 1) - curve(i, 1)) * std::copysign(1.0, dx);
            dx = std::abs(dx);
            decomp.compute(param_mat.middleCols(i - ifitbegin, n_fitpts));
            circ_coeffs = decomp.transpose().solve(param_vec.segment(i - ifitbegin, n_fitpts));  // Note transpose here
            //circ_coeffs(0) /= -2.0;  // Center coordinates of the circle
            //circ_coeffs(1) /= -2.0;
            //curvature_(i) = std::copysign(1.0 / std::sqrt(circ_coeffs(0) * circ_coeffs(0) + circ_coeffs(1) * circ_coeffs(1) - circ_coeffs(2)),
            //                              (curve(i, 0) - circ_coeffs(0)) * dy - (curve(i, 1) - circ_coeffs(1)) * dx);
            centerx = -circ_coeffs(1) / (2.0 * circ_coeffs(0));  // Center coordinates of the circle
            centery = -circ_coeffs(2) / (2.0 * circ_coeffs(0));
            curvature_(i) = std::copysign(2.0 * std::abs(circ_coeffs(0))
                                          / std::sqrt(circ_coeffs(1) * circ_coeffs(1) + circ_coeffs(2) * circ_coeffs(2) - 4.0 * circ_coeffs(0)),
                                          (curve(i, 0) - centerx) * dy - (curve(i, 1) - centery) * dx);
        }
    }
}

// Fit curve to Fermi-Dirac function, f(x, a) = a0 / (1 + exp(a2 * x + a3)) + a1, which has 4 parameters.
template <int _n0, int _n1, int _nm>
template <typename Derived, typename OtherDerived>
Eigen::Vector<typename Derived::Scalar, 4> MQEMContinuator<_n0, _n1, _nm>::fitFDFunc(const Eigen::DenseBase<Derived> &curve, const Eigen::DenseBase<OtherDerived>& fitted) {
    typedef typename Derived::Scalar Scalar;
    Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 4> Jac(curve.rows(), 4);
    Eigen::Vector<Scalar, Derived::RowsAtCompileTime> dy(curve.rows());
    Eigen::Vector<Scalar, 4> a, da;
    const auto max_iter = std::any_cast<Eigen::Index>(parameters.at("FDfit_max_iteraction"));
    const auto tol = std::any_cast<double>(parameters.at("FDfit_tolerance"));
    const auto damp = std::any_cast<double>(parameters.at("FDfit_damp"));
    Eigen::Index iter = 0;
    Scalar tmp;
    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 4> > decomp(curve.rows(), 4);
    
    // Initialize fitting parameter
    a(0) = curve(Eigen::last, 1) - curve(0, 1);
    a(1) = curve(0, 1);
    a(2) = 12.0 / (curve(0, 0) - curve(Eigen::last, 0));
    a(3) = -6.0 * (curve(0, 0) + curve(Eigen::last, 0)) / (curve(0, 0) - curve(Eigen::last, 0));
    while (true) {
        for (Eigen::Index i = 0; i < Jac.rows(); ++i) {
            // Assemble Jacobian matrix df/da
            tmp = std::exp(a(2) * curve(i, 0) + a(3));
            Jac(i, 0) = 1.0 / (1.0 + tmp);
            Jac(i, 1) = 1.0;
            Jac(i, 2) = -a(0) * Jac(i, 0) * Jac(i, 0) * tmp * curve(i, 0);
            Jac(i, 3) = -a(0) * Jac(i, 0) * Jac(i, 0) * tmp;
            // Assemble residue vector
            dy(i) = curve(i, 1) - a(0) * Jac(i, 0) - a(1);
        }
        // Solve normal equations
        decomp.compute(Jac);
        da = decomp.solve(dy);
        
        if (!Jac.array().isFinite().all() || !dy.array().isFinite().all() || !da.array().isFinite().all())
            throw std::runtime_error("MQEMContinuator::fitFDFunc: fitting diverged");
        else if ((da.array().abs() < tol).all()) break;
        else if (iter == max_iter) throw std::runtime_error("MQEMContinuator::fitFDFunc: fitting not converged within maximum iterations");
        
        a += damp * da;
        ++iter;
    }
    
    Eigen::DenseBase<OtherDerived>& fitted_ = const_cast<Eigen::DenseBase<OtherDerived>&>(fitted);  // Make curvature writable by casting away its const-ness
    fitted_.derived().resize(curve.rows(), 2);  // Resize the derived object
    
    // Calculate second derivative of fitted Fermi-Dirac function
    for (Eigen::Index i = 0; i < fitted_.rows(); ++i) {
        tmp = std::exp(a(2) * curve(i, 0) + a(3));
        fitted_(i, 0) = a(0) / (1.0 + tmp) + a(1);  // Fitted FD function
        fitted_(i, 1) = a(0) * a(2) * a(2) * tmp * (tmp - 1.0) / ((1.0 + tmp) * (1.0 + tmp) * (1.0 + tmp));  // Second derivative
        tmp = -a(0) * a(2) * tmp / ((1.0 + tmp) * (1.0 + tmp));  // First derivative
        tmp = std::sqrt(1.0 + tmp * tmp);
        fitted_(i, 1) /= tmp * tmp * tmp;  // Curvature
    }
    
    return a;
}

// Fit curve to cubic spline with not-a-knot boundary condition; first row of deriv_curv stores fitted first derivative, second row stores fitted curvature
template <int _n0, int _n1, int _nm>
template <typename Derived, typename OtherDerived>
void MQEMContinuator<_n0, _n1, _nm>::fitCubicSpline(const Eigen::DenseBase<Derived> &curve, const Eigen::DenseBase<OtherDerived> &deriv_curv) {
    const Eigen::Index np = curve.rows();
    if (np < 3) throw std::range_error("MQEMContinuator::fitCubicSpline: #points to fit is less than 3");
    
    typedef typename Derived::Scalar Scalar;
    // Coefficient matrix, always real; At the beginning fill the diagonal part
    Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::RowsAtCompileTime> A =
    2.0 * Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::RowsAtCompileTime>::Identity(np, np);
    // Vector for the rhs values
    Eigen::Vector<Scalar, Derived::RowsAtCompileTime> b(np);
    
    // Construct the coefficient matrix and vector of the linear equations for the second derivative of the cubic spline S"(x)
    // First row, for the boundary condition y'''_1(x_1) = y'''_2(x_1), i.e., (M_1 - M_0) / h_1 = (M_2 - M_1) / h_2
    // => -M_0 + (h_2 + h_1) / h_2 * M_1 - h_1 / h_2 * M_2 = 0
    A(0, 0) = -1.0;
    A(0, 1) = (curve(2, 0) - curve(0, 0)) / (curve(2, 0) - curve(1, 0));
    A(0, 2) = -(curve(1, 0) - curve(0, 0)) / (curve(2, 0) - curve(1, 0));
    b(0) = 0.0;
    // Middle part
    for (Eigen::Index i = 1; i < np - 1; ++i) {
        A(i, i - 1) = (curve(i, 0) - curve(i - 1, 0)) / (curve(i + 1, 0) - curve(i - 1, 0));
        A(i, i + 1) = 1.0 - A(i, i - 1);
        b(i) = 6.0 / (curve(i + 1, 0) - curve(i - 1, 0)) * ((curve(i + 1, 1) - curve(i, 1)) / (curve(i + 1, 0) - curve(i, 0))
                                                            - (curve(i, 1) - curve(i - 1, 1)) / (curve(i, 0) - curve(i - 1, 0)));
    }
    // Last row, for the boundary condition y'''_(n-1)(x_(n-1)) = y'''_n(x_(n-1)), i.e., (M_(n-1) - M_(n-2)) / h_(n-1) = (M_n - M_(n-1)) / h_n
    // => -M_(n-2) + (h_n + h_(n-1)) / h_n * M_(n-1) - h_(n-1) / h_n * M_n = 0 (n = np - 1)
    A(np - 1, np - 3) = -1.0;
    A(np - 1, np - 2) = (curve(np - 1, 0) - curve(np - 3, 0)) / (curve(np - 1, 0) - curve(np - 2, 0));
    A(np - 1, np - 1) = (curve(np - 2, 0) - curve(np - 3, 0)) / (curve(np - 1, 0) - curve(np - 2, 0));
    b(np - 1) = 0.0;
    
    Eigen::DenseBase<OtherDerived>& deriv_curv_ = const_cast<Eigen::DenseBase<OtherDerived>&>(deriv_curv);  // Make curvature writable by casting away its const-ness
    deriv_curv_.derived().resize(np, 2);  // Resize the derived object
    
    // Solve for the second derivatives
    Eigen::ColPivHouseholderQR<Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::RowsAtCompileTime> > dec(A);
    deriv_curv_.col(1) = dec.solve(b);
    
    // Calculate first derivatives
    deriv_curv_(0, 0) = deriv_curv_(1, 1) * (curve(1, 0) - curve(0, 0)) / 2.0 + (curve(1, 1) - curve(0, 1)) / (curve(1, 0) - curve(0, 0))
    - (deriv_curv_(1, 1) - deriv_curv_(0, 1)) * (curve(1, 0) - curve(0, 0)) / 6.0;
    for (Eigen::Index i = 1; i < np; ++i) deriv_curv_(i, 0) = deriv_curv_(i, 1) * (curve(i, 0) - curve(i - 1, 0)) / 2.0
        + (curve(i, 1) - curve(i - 1, 1)) / (curve(i, 0) - curve(i - 1, 0))
        - (deriv_curv_(i, 1) - deriv_curv_(i - 1, 1)) * (curve(i, 0) - curve(i - 1, 0)) / 6.0;
    deriv_curv_.col(1).array() /= (1.0 + deriv_curv_.col(0).array().square()).sqrt().cube();  // This is curvature
}

typedef MQEMContinuator<2, Eigen::Dynamic, Eigen::Dynamic> MQEMContinuator2XX;

#endif /* mqem_hpp */
