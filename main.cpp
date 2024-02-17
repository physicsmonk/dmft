//
//  main.cpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#include <thread>       // std::this_thread::sleep_for
#include <chrono>
#include <limits>
#include "hdf5.h"
#include "input.hpp"
#include "ct_aux_imp_solver.hpp"
#include "self_consistency.hpp"
//#include "pade.hpp"
#include "mqem.hpp"

//#ifdef _WIN32
//#include <Windows.h>
//#else
//#include <unistd.h>
//#endif

using namespace std::complex_literals;



// Derive from the BareHamiltonian class the user-defined Hamiltonian
class MultilayerInMag : public BareHamiltonian {
public:
    int p, q;  // Constitutes the magnetic field
    
    void constructHamiltonian(const Eigen::VectorXd& k, Eigen::MatrixXcd& H) const override {  // Only need to construct the lower triangular part
        if (q < 1) throw std::range_error("q cannot be less than 1!");
        std::complex<double> en;
        H = Eigen::MatrixXcd::Zero(m_nsites * q, m_nsites * q);
        
        // Construct each layer
        for (int m = 0; m < q; ++m) {
            en = 2.0 * m_t(0) * cos(k[1] + (2.0 * M_PI * p) / q * m);
            for (int l = 0; l < m_nsites; ++l) {
                H(m + l * q, m + l * q) = en;
                if (m < q - 1) H(m + l * q + 1, m + l * q) = m_t(0);
            }
        }
        const std::complex<double> hop = m_t(0) * std::exp(1i * std::fmod(k[0] * q, 2 * M_PI));
        for (int l = 0; l < m_nsites; ++l) {
            H(q - 1 + l * q, l * q) += hop;  // "+=" works for all q > 0
            H(l * q, q - 1 + l * q) += std::conj(hop);  // This only plays a role in q = 1 case because we only use the lower triangular part of H
        }
        // Construct interlayer coupling
        for (int l = 0; l < m_nsites - 1; ++l) H(Eigen::seqN((l + 1) * q, q), Eigen::seqN(l * q, q)).diagonal().setConstant(m_t(1));
    }
    
    // Only need to construct the lower triangular part
    void constructFermiVelocity(const int coord, const Eigen::VectorXd& k, Eigen::MatrixXcd& v) const override {
        if (coord < 0 || coord > 1) throw std::range_error("Coordinate can only be 0 or 1 for 2D multilayer Hubbard model in magnetic fields!");
        if (q < 1) throw std::range_error("q cannot be less than 1!");
        
        std::complex<double> v0;
        v = Eigen::MatrixXcd::Zero(m_nsites * q, m_nsites * q);
        // Note the Fermi velocity matrix is already block diagonal; the spaces of each layer are decoupled.
        if (coord == 0) {  // Fermi velocity in x direction, along which the magnetic unit cell expands
            for (int l = 0; l < m_nsites; ++l) {
                for (int m = 0; m < q - 1; ++m) v(m + 1 + l * q, m + l * q) = -1i * m_t(0);
            }
            const std::complex<double> hop = m_t(0) * std::exp(1i * std::fmod(k[0] * q, 2 * M_PI));
            v0 = 1i * static_cast<double>(q) * hop - 1i * static_cast<double>(q - 1) * hop;
            for (int l = 0; l < m_nsites; ++l) {
                v(q - 1 + l * q, l * q) += v0;  // "+=" works for all q > 0
                v(l * q, q - 1 + l * q) += std::conj(v0);  // This only plays a role in q = 1 case because we only use the lower triangular part of H
            }
        }
        else if (coord == 1) {  // Fermi velocity in y direction
            for (int m = 0; m < q; ++m) {
                v0 = -2.0 * m_t(0) * sin(k[1] + (2.0 * M_PI * p) / q * m);
                for (int l = 0; l < m_nsites; ++l) v(m + l * q, m + l * q) = v0;
            }
        }
    }
    
    // Second derivative of noninteracting k-dependent energy at zero magnetic field
    void constructBandCurvature(const int co1, const int co2, const Eigen::VectorXd& k, Eigen::MatrixXcd& eps12) const override {
        if (co1 < 0 || co1 > 1 || co2 < 0 || co2 > 1) throw std::range_error("Coordinate can only be 0 or 1 for 2D multilayer Hubbard model in magnetic fields!");
        eps12 = Eigen::MatrixXcd::Zero(2, 2);
        if (co1 == co2) {
            eps12(0, 0) = -2.0 * m_t(0) * std::cos(k[co1]);
            eps12(1, 1) = eps12(0, 0);
        }
    }
};

// 3D Hubbard model
//class Hubbard : public BareHamiltonian {
//    void constructHamiltonian(const Eigen::VectorXd& k, Eigen::MatrixXcd& H) const {
//        const double t = 1.0;
//        const double a = 1.0;
//        H.resize(1, 1);
//        H << -2 * t * cos(k[0] * a) - 2 * t * cos(k[1] * a) - 2 * t * cos(k[2] * a);
//    }
//};

//struct cut {
//    Eigen::Index size() const {return shrunk_size;}
//    Eigen::Index operator[](Eigen::Index i) const {return i < cut_ind ? i : i + 1;}
//    Eigen::Index shrunk_size, cut_ind;
//};







int main(int argc, char * argv[]) {
    // Get starting timepoint
    auto start = std::chrono::high_resolution_clock::now();
    
    int psize, prank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    
    // For test
    //Eigen::RowVector4i aa;
    //aa << 3, 0, 1, 2;
    //Eigen::PermutationMatrix<4> perm;
    //perm.indices() = {1, 2, 3, 0};
    //std::cout << aa * perm << std::endl;
    //Eigen::MatrixXi A(3, 3);
    //A.reshaped() = Eigen::VectorXi::LinSpaced(9, 1, 9);
    //std::cout << A << std::endl;
    //std::cout << A(1, cut{2, 1}) << std::endl;
    //A(cut{2, 1}, cut{2, 1}) += Eigen::Matrix2i::Constant(10);
    //A = A(cut{2, 1}, cut{2, 1}).eval();
    //std::cout << A << std::endl;
    //std::istringstream in;
    //in.str("1.1 2.2\n3.3 4.4");
    //std::string word;
    //in >> word;
    //Eigen::ArrayXXd arr;
    //in >> arr;
    //std::cout << arr << std::endl;
    
    /*
    Eigen::Index Nul = 5, Nur = 5, Nw = 51, Niw = 71;
    double omegal = -3.0, omegar = 3.0, invT = 10.0;
    Eigen::ArrayXd iomegas = Eigen::ArrayXd::LinSpaced(Niw, M_PI / invT, (2 * Niw - 1) * M_PI / invT);
    MQEMContinuator<1, Eigen::Dynamic, 2> testmqem;
    //mqem.parameters.at("Gaussian_sigma") = 4.0;
    testmqem.assembleKernelMatrix(iomegas, Nul, omegal, Nw, omegar, Nur);
    
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, 2> spec(1, testmqem.realFreqGrid().size(), 2), func_mats(1, Niw, 2, MPI_COMM_WORLD);
    Eigen::Matrix2cd rot;
    double theta;
    // Assign model spectral function
    for (Eigen::Index i = 0; i < testmqem.realFreqGrid().size(); ++i) {
        theta = 2.0 * M_PI * testmqem.realFreqGrid()(i) * 0.05; theta *= theta;
        rot << std::cos(theta), 1i * std::sin(theta),
               1i * std::sin(theta), std::cos(theta);
        spec[i] << 0.5 * M_2_SQRTPI * M_SQRT1_2 * (std::exp(-0.5 * (testmqem.realFreqGrid()(i) + 1.5) * (testmqem.realFreqGrid()(i) + 1.5))
        + std::exp(-0.5 * (testmqem.realFreqGrid()(i) - 1.5) * (testmqem.realFreqGrid()(i) - 1.5))), 0.0, 0.0, 0.0;
        //std::cout << spec[i] << std::endl;
        spec[i] = rot.adjoint() * spec[i] * rot;
    }
    SqMatArray<std::complex<double>, 1, 1, 2> mom;
    mom.dim1RowVecsAtDim0(0).noalias() = spec.dim1RowVecsAtDim0(0) * testmqem.realFreqIntVector();
    
    // Recover Matsubara function
    func_mats.dim1RowVecsAtDim0(0).transpose().noalias() = testmqem.kernelMatrix() * spec.dim1RowVecsAtDim0(0).transpose();
    // Add noise to Matsubara function
    std::default_random_engine gen;
    double sig = 1e-3;
    SqMatArray<double, 1, Eigen::Dynamic, 2> vars(1, Niw, 2, MPI_COMM_WORLD);
    vars().setConstant(sig * sig * 2.0);
    
    std::normal_distribution<double> nrd(0.0, sig);
    for (Eigen::Index i = 0; i < Niw; ++i) {
        rot << nrd(gen) + 1i * nrd(gen), nrd(gen) + 1i * nrd(gen), nrd(gen) + 1i * nrd(gen), nrd(gen) + 1i * nrd(gen);
        func_mats[i] += rot;
    }
    
    bool cvg;
    cvg = testmqem.computeSpectra(iomegas, func_mats, vars, mom);
    testmqem.computeRetardedFunc();
    
    if (prank == 0) {
        printData("real_freqs.txt", testmqem.realFreqGrid());
        printData("model_spec_func.txt", spec);
        printData("model_mats_func.txt", func_mats);
        std::cout << cvg << std::endl;
        printData("default_model.txt", testmqem.defaultModel());
        printData("spec_func.txt", testmqem.spectra());
        printData("log10chi2_log10alpha.txt", testmqem.log10chi2Log10alpha(0));
        printData("retarded_func.txt", testmqem.retardedFunc());
    }
    */
    
    //MPI_Finalize();
    //return 0;
    
    
    // Prepare for hdf5 output
    hid_t complex_id = 0;
    hid_t file_id = 0;
    hid_t grp_g0_id = 0, grp_gf_id = 0, grp_se_id = 0, grp_sp_id = 0, grp_id = 0;
    hid_t dspace_t_id = 0, dspace_iw_id = 0, dspace_w_id = 0, dspace_id = 0;
    hid_t dset_g0_t_id = 0, dset_g0_iw_id = 0, dset_gf_t_id = 0, dset_gf_iw_id = 0, dset_se_dyn_id = 0,
          dset_se_static_id = 0, dset_se_var_id = 0, dset_se_mom_id = 0, dset_se_ret_id = 0,
          dset_se_tail_id = 0, dset_se_dm_id = 0, dset_sp_w_id = 0, dset_sp_k0_id = 0, dset_sp_kw_id = 0,
          dset_id = 0;
    hid_t attr_cond_id = 0;
    herr_t status;
    constexpr int RANK_MAT = 2;
    constexpr int RANK_GF = 4;
    constexpr int RANK_STATIC = 3;
    constexpr int RANK_SP = 5;
    hsize_t dims_mat[RANK_MAT], dims_gf[RANK_GF], dims_static[RANK_STATIC], dims_sp[RANK_SP];
    
    // Create complex type for hdf5 compatible with h5py
    if (prank == 0) {
        complex_id = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
        status = H5Tinsert(complex_id, "r", 0, H5T_NATIVE_DOUBLE);
        status = H5Tinsert(complex_id, "i", sizeof(double), H5T_NATIVE_DOUBLE);
    }
    
    
    std::string sep("----------------------------------------------------------------------");
    
    // std::random_device rd;
    // std::cout << rd() << std::endl;
    
    Eigen::Index nsites = 2;
    double t = -1.0;
    double tz = -1.0;
    int q = 2;
    int p = 1;
    double U = 4.0;
    double beta = 10.0;
    // mu_eff is the effective chemical potential of the interacting system for the impurity solver while mu = mu_eff + U / 2 is
    // the true chemical potential. mu_eff = 0 corresponds to half-filling, which is default.
    double mu_eff = 0.0;
    // Goal electron density; negative means to fix chemical potential while positive means to fix electron density to this goal value,
    // in which case mu_eff is taken as initial value
    double density_goal = -1.0;
    double K = 1.0;
    
    Eigen::Array<Eigen::Index, 2, 1> nk;
    nk << 101, 101;
    Eigen::Index nbins4dos = 1001;

    Eigen::Index nfcut = 1000;   // 1000
    Eigen::Index ntau = 301;  // 1001
    Eigen::Index ntau4eiwt = 0;
    Eigen::Index nbins4S = 10001;   // 10001

    Eigen::Index markovchainlength = 100000000;    // 20000000
    double qmctimelimit = 8.0;  // Unit is minute
    Eigen::Index measureperiod = 200;    // 200
    Eigen::Index warmupsize = 10000;
    std::string measurewhat("S");
    Eigen::Index histmaxorder = 100;
    std::string magneticorder("paramagnetic");
    std::string verbosity("off");
    
    std::string ansatz("metal");
    int nitmax = 20;
    double G0stepsize = 1.0;
    //std::string converge_type("Gimp_Glat_max_error");
    double converge_criterion = 0.005;
    double density_error = 0.001;
    Eigen::Index tailstart = 100;
    
    /*
    // For Pade interpolation
    bool physonly = true;
    int ndatalens = 13;
    int mindatalen = 50;
    int maxdatalen = 98;
    int nstartfreqs = 1;
    int minstartfreq = 0;
    int maxstartfreq = 0;
    int ncoefflens = 13;
    int mincoefflen = 10;
    int maxcoefflen = 34;
    Eigen::Index nenergies = 201;
    double minenergy = -10.0;
    double maxenergy = 10.0;
    double delenergy = 0.01;
    int mpprec = 256;
     */
    
    // For MQEM analytic continuation
    bool simple_default_model = false;
    double secant_tol = 0.001;
    Eigen::Index secant_maxiter = 30;
    double secant_damp = 0.1;
    double mat_inv_threshold = -1.0;
    Eigen::Index n_lrealfreq = 10;
    Eigen::Array<double, 7, 1> midrealfreq_anchors_steps;
    midrealfreq_anchors_steps << -5.0, 0.1, -2.0, 0.05, 2.0, 0.1, 5.0;
    Eigen::Index n_rrealfreq = 10;
    double pulay_mix = 0.01;
    Eigen::Index pulay_histsize = 5;
    Eigen::Index pulay_period = 3;
    double pulay_tol = 1e-6;
    Eigen::Index pulay_maxiter = 500;
    //double gaussian_sig = 1.0;
    double alpha_maxfac = 100.0;
    double alpha_infofitfac = 0.05;
    double alpha_initfrac = 0.01;
    double alpha_stopslope = 0.01;  // Negative to actually not use this criterion
    double alpha_stopstep = 1e-5;
    double alpha_dAtol = 0.1;
    double alpha_rmin = 0.7;
    double alpha_rmax = 2.0;
    double alpha_rscale = 0.8;
    std::string curvfit_method("FD");
    Eigen::Index arcfit_size = 9;
    Eigen::Index alpha_capacity = 1000;
    bool alpha_cacheall = true;
    double fdfit_damp = 0.1;
    double fdfit_tol = 1e-4;
    Eigen::Index fdfit_maxiter = 500;
    double pint_eps = 1e-6;
    
    int proc_control = 0;
    bool computesigmaxy = true;
    int n_computecond = 1;   // Number of times computing conductivities after convergence
    int intalg = CubicSpline;
    
    bool loc_corr = false;   // Whether to use local correlation approximation (off-diagonal elements of self-energy are zero)


    pugi::xml_document doc;
    pugi::xml_node docroot;
    if (!doc.load_file("input.xml")) {
        std::cout << "Error: Cannot load file input.xml" << std::endl;
        return -1;
    }
    docroot = doc.child("input");

    readxml_bcast(nsites, docroot, "physical/numSites", MPI_COMM_WORLD);
    readxml_bcast(t, docroot, "physical/hoppingXy", MPI_COMM_WORLD);
    readxml_bcast(tz, docroot, "physical/hoppingXy.hoppingZ", MPI_COMM_WORLD);
    readxml_bcast(q, docroot, "physical/q", MPI_COMM_WORLD);
    readxml_bcast(p, docroot, "physical/q.p", MPI_COMM_WORLD);
    readxml_bcast(beta, docroot, "physical/inverseTemperature", MPI_COMM_WORLD);
    readxml_bcast(U, docroot, "physical/onsiteCoulombRepulsion", MPI_COMM_WORLD);
    readxml_bcast(mu_eff, docroot, "physical/effectiveChemicalPotential", MPI_COMM_WORLD);
    readxml_bcast(density_goal, docroot, "physical/density", MPI_COMM_WORLD);
    readxml_bcast(nbins4dos, docroot, "numerical/bareDOS/numBins", MPI_COMM_WORLD);
    readxml_bcast(nk(0), docroot, "numerical/bareDOS/numBins.numkx", MPI_COMM_WORLD);
    readxml_bcast(nk(1), docroot, "numerical/bareDOS/numBins.numky", MPI_COMM_WORLD);
    readxml_bcast(nfcut, docroot, "numerical/GreenFunction/frequencyCutoff", MPI_COMM_WORLD);
    readxml_bcast(ntau, docroot, "numerical/GreenFunction/tauGridSize", MPI_COMM_WORLD, prank);
    readxml_bcast(ntau4eiwt, docroot, "numerical/GreenFunction/tauGridSizeOfExpiwt", MPI_COMM_WORLD);
    readxml_bcast(nbins4S, docroot, "numerical/GreenFunction/numTauBinsForSelfEnergy", MPI_COMM_WORLD);
    readxml_bcast(markovchainlength, docroot, "numerical/QMC/MarkovChainLength", MPI_COMM_WORLD);
    readxml_bcast(qmctimelimit, docroot, "numerical/QMC/timeLimit", MPI_COMM_WORLD);
    readxml_bcast(measureperiod, docroot, "numerical/QMC/measurePeriod", MPI_COMM_WORLD);
    readxml_bcast(warmupsize, docroot, "numerical/QMC/numWarmupSteps", MPI_COMM_WORLD);
    readxml_bcast(measurewhat, docroot, "numerical/QMC/whatToMeasure", MPI_COMM_WORLD);
    readxml_bcast(histmaxorder, docroot, "numerical/QMC/maxOrderForHistogram", MPI_COMM_WORLD);
    readxml_bcast(magneticorder, docroot, "numerical/QMC/magneticOrder", MPI_COMM_WORLD);
    readxml_bcast(verbosity, docroot, "numerical/QMC/verbosity", MPI_COMM_WORLD);
    readxml_bcast(ansatz, docroot, "numerical/selfConsistency/ansatz", MPI_COMM_WORLD);
    readxml_bcast(nitmax, docroot, "numerical/selfConsistency/maxIteration", MPI_COMM_WORLD);
    readxml_bcast(G0stepsize, docroot, "numerical/selfConsistency/stepSizeForUpdateG0", MPI_COMM_WORLD);
    //readxml_bcast(converge_type, docroot, "numerical/selfConsistency/convergeType", MPI_COMM_WORLD);
    readxml_bcast(converge_criterion, docroot, "numerical/selfConsistency/convergeCriterion", MPI_COMM_WORLD);
    readxml_bcast(density_error, docroot, "numerical/selfConsistency/densityError", MPI_COMM_WORLD);
    readxml_bcast(tailstart, docroot, "numerical/selfConsistency/selfEnergyTailStartIndex", MPI_COMM_WORLD);
    /*
    readxml_bcast(physonly, docroot, "numerical/PadeInterpolation/physicalSpectraOnly", MPI_COMM_WORLD, prank);
    readxml_bcast(ndatalens, docroot, "numerical/PadeInterpolation/numDataLengths", MPI_COMM_WORLD, prank);
    readxml_bcast(mindatalen, docroot, "numerical/PadeInterpolation/numDataLengths.min", MPI_COMM_WORLD, prank);
    readxml_bcast(maxdatalen, docroot, "numerical/PadeInterpolation/numDataLengths.max", MPI_COMM_WORLD, prank);
    readxml_bcast(nstartfreqs, docroot, "numerical/PadeInterpolation/numStartFrequencies", MPI_COMM_WORLD, prank);
    readxml_bcast(minstartfreq, docroot, "numerical/PadeInterpolation/numStartFrequencies.min", MPI_COMM_WORLD, prank);
    readxml_bcast(maxstartfreq, docroot, "numerical/PadeInterpolation/numStartFrequencies.max", MPI_COMM_WORLD, prank);
    readxml_bcast(ncoefflens, docroot, "numerical/PadeInterpolation/numCoefficientLengths", MPI_COMM_WORLD, prank);
    readxml_bcast(mincoefflen, docroot, "numerical/PadeInterpolation/numCoefficientLengths.min", MPI_COMM_WORLD, prank);
    readxml_bcast(maxcoefflen, docroot, "numerical/PadeInterpolation/numCoefficientLengths.max", MPI_COMM_WORLD, prank);
    readxml_bcast(nenergies, docroot, "numerical/PadeInterpolation/energyGridSize", MPI_COMM_WORLD, prank);
    readxml_bcast(minenergy, docroot, "numerical/PadeInterpolation/energyGridSize.min", MPI_COMM_WORLD, prank);
    readxml_bcast(maxenergy, docroot, "numerical/PadeInterpolation/energyGridSize.max", MPI_COMM_WORLD, prank);
    readxml_bcast(delenergy, docroot, "numerical/PadeInterpolation/energyGridSize.delta", MPI_COMM_WORLD, prank);
    readxml_bcast(mpprec, docroot, "numerical/PadeInterpolation/internalPrecision", MPI_COMM_WORLD, prank);
    */
    readxml_bcast(simple_default_model, docroot, "numerical/MQEM/simpleDefaultModel", MPI_COMM_WORLD);
    readxml_bcast(secant_maxiter, docroot, "numerical/MQEM/secantMaxIteration", MPI_COMM_WORLD);
    readxml_bcast(secant_tol, docroot, "numerical/MQEM/secantTolerance", MPI_COMM_WORLD);
    readxml_bcast(secant_damp, docroot, "numerical/MQEM/secantDamp", MPI_COMM_WORLD);
    readxml_bcast(mat_inv_threshold, docroot, "numerical/MQEM/matrixInvertibleThreshold", MPI_COMM_WORLD);
    readxml_bcast(n_lrealfreq, docroot, "numerical/MQEM/numLeftRealFrequencies", MPI_COMM_WORLD);
    std::string rfai;
    std::stringstream rfai_;
    readxml_bcast(rfai, docroot, "numerical/MQEM/midRealFrequencyAnchorsSteps", MPI_COMM_WORLD);
    rfai_ << rfai;
    rfai_ >> midrealfreq_anchors_steps;
    readxml_bcast(n_rrealfreq, docroot, "numerical/MQEM/numRightRealFrequencies", MPI_COMM_WORLD);
    readxml_bcast(pulay_mix, docroot, "numerical/MQEM/PulayMixingParameter", MPI_COMM_WORLD);
    readxml_bcast(pulay_histsize, docroot, "numerical/MQEM/PulayHistorySize", MPI_COMM_WORLD);
    readxml_bcast(pulay_period, docroot, "numerical/MQEM/PulayPeriod", MPI_COMM_WORLD);
    readxml_bcast(pulay_tol, docroot, "numerical/MQEM/PulayTolerance", MPI_COMM_WORLD);
    readxml_bcast(pulay_maxiter, docroot, "numerical/MQEM/PulayMaxIteration", MPI_COMM_WORLD);
    //readxml_bcast(gaussian_sig, docroot, "numerical/MQEM/defaultModelSigma", MPI_COMM_WORLD);
    readxml_bcast(alpha_maxfac, docroot, "numerical/MQEM/alphaMaxFactor", MPI_COMM_WORLD);
    readxml_bcast(alpha_infofitfac, docroot, "numerical/MQEM/alphaInfoFitFactor", MPI_COMM_WORLD);
    readxml_bcast(alpha_initfrac, docroot, "numerical/MQEM/alphaInitFraction", MPI_COMM_WORLD);
    readxml_bcast(alpha_stopslope, docroot, "numerical/MQEM/alphaStopSlope", MPI_COMM_WORLD);
    readxml_bcast(alpha_stopstep, docroot, "numerical/MQEM/alphaStopStep", MPI_COMM_WORLD);
    readxml_bcast(alpha_dAtol, docroot, "numerical/MQEM/alphaSpectrumRelativeError", MPI_COMM_WORLD);
    readxml_bcast(alpha_rmin, docroot, "numerical/MQEM/alphaStepMinRatio", MPI_COMM_WORLD);
    readxml_bcast(alpha_rmax, docroot, "numerical/MQEM/alphaStepMaxRatio", MPI_COMM_WORLD);
    readxml_bcast(alpha_rscale, docroot, "numerical/MQEM/alphaStepScale", MPI_COMM_WORLD);
    readxml_bcast(alpha_capacity, docroot, "numerical/MQEM/alphaCapacity", MPI_COMM_WORLD);
    readxml_bcast(alpha_cacheall, docroot, "numerical/MQEM/alphaCacheAll", MPI_COMM_WORLD);
    readxml_bcast(curvfit_method, docroot, "numerical/MQEM/curvatureFitMethod", MPI_COMM_WORLD);
    readxml_bcast(arcfit_size, docroot, "numerical/MQEM/arcFitSize", MPI_COMM_WORLD);
    readxml_bcast(fdfit_damp, docroot, "numerical/MQEM/FDFitDamp", MPI_COMM_WORLD);
    readxml_bcast(fdfit_tol, docroot, "numerical/MQEM/FDFitTolerance", MPI_COMM_WORLD);
    readxml_bcast(fdfit_maxiter, docroot, "numerical/MQEM/FDFitMaxIteraction", MPI_COMM_WORLD);
    readxml_bcast(pint_eps, docroot, "numerical/MQEM/principalIntEps", MPI_COMM_WORLD);
    readxml_bcast(proc_control, docroot, "processControl/generalProcess", MPI_COMM_WORLD);
    readxml_bcast(computesigmaxy, docroot, "processControl/computeHallConductivity", MPI_COMM_WORLD);
    readxml_bcast(n_computecond, docroot, "processControl/numComputeConductivity", MPI_COMM_WORLD);
    readxml_bcast(intalg, docroot, "processControl/integrationAlgorithm", MPI_COMM_WORLD);
    IntAlg intalg_ = static_cast<IntAlg>(intalg);
    readxml_bcast(loc_corr, docroot, "processControl/localCorrelation", MPI_COMM_WORLD);

    if (prank == 0) std::cout << sep << std::endl;
    
    // Setup bare Hamiltonian
    auto H0 = std::make_shared<MultilayerInMag>();
    H0->setMPIcomm(MPI_COMM_WORLD);
    
    H0->numSites(nsites);
    H0->hopMatElem(Eigen::Array2cd(t, tz));
    //std::cout << H0->hopMatElem(0) << ", " << H0->hopMatElem(1) << std::endl;
    H0->q = q;
    H0->p = p;
    
    H0->chemPot(mu_eff);   // Chemical potential of the noninterating system (in comparison with the interacting system) is formally the effective one.
    
    // Note this constructor: if nc is of type int, on some computers it could be implicitely converted to MPI_Comm type, not Eigen::Index type,
    // and call another constructor, very dangerous
    //SqMatArray22Xcd moments(2, 2, nsites);
    H0->moments().resize(2, 2, nsites);
    H0->moments()().setZero();
    for (int s = 0; s < 2; ++s) {
        for (int l = 0; l < nsites; ++l) {
            if (l < nsites - 1) {
                H0->moments()(s, 0, l + 1, l) = tz;
                H0->moments()(s, 0, l, l + 1) = std::conj(tz);
            }
            H0->moments()(s, 1, l, l) = 4.0 * t * std::conj(t) + tz * std::conj(tz);
            if (l < nsites - 2) {
                H0->moments()(s, 1, l + 2, l) = tz * tz;
                H0->moments()(s, 1, l, l + 2) = std::conj(H0->moments()(s, 1, l + 2, l));
            }
        }
    }
    
    H0->primVecs((Eigen::Matrix2d() << q, 0, 0, 1).finished());
    
    H0->type("multilayer_in_mag");   // general, bethe, bethe_dimer, multilayer_in_mag
    
    const Eigen::Index nkmin = nk.minCoeff();
    Eigen::Array<Eigen::Index, 2, Eigen::Dynamic> kidpath(2, nk(0) / 2 + nk(1) / 2 + nkmin / 2 + 3);
    for (Eigen::Index i = 0; i <= nk(0) / 2; ++i) kidpath.col(i) << nk(0) / 2 - i, nk(1) / 2;  // Gamma -> X
    for (Eigen::Index i = 0; i <= nk(1) / 2; ++i) kidpath.col(nk(0) / 2 + 1 + i) << 0, nk(1) / 2 - i;  // X -> M
    for (Eigen::Index i = 0; i <= nkmin / 2; ++i) kidpath.col(nk(0) / 2 + nk(1) / 2 + 2 + i) << i * nk(0) / nkmin, i * nk(1) / nkmin;  // M -> Gamma
    
    H0->computeBands(nk, kidpath);
    H0->computeDOS(nbins4dos);
    
//    std::array<double, 2> erange = {-2.0 * std::fabs(t), 2.0 * std::fabs(t)};
//    Eigen::ArrayXd semicircle(nbins4dos);
//    double energy;
//    for (Eigen::Index ie = 0; ie < semicircle.size(); ++ie) {
//        energy = (ie + 0.5) * (erange[1] - erange[0]) / semicircle.size() + erange[0];
//        semicircle[ie] = sqrt(4.0 * t * t - energy * energy) / (2 * M_PI * t * t);
//    }
    // H0->setDOS(erange, semicircle);
    // H0->setDOS(erange, semicircle);
//    H0->setDOS(erange, semicircle);
    
    
    // Test
    // MPI_Finalize();
    // return 0;
    
    /*
    // Set up decoupled Hamiltonian for the local correlation approximation
    auto H0dec = std::make_shared<Dimer2DinMag>();
    if (loc_corr) {
        H0dec->setMPIcomm(MPI_COMM_WORLD);
        
        H0dec->hopMatElem(Eigen::Array2cd(t, 0.0));
        H0dec->q = q;
        H0dec->p = p;
        
        H0dec->chemPot(mu_eff);   // Chemical potential of the noninterating system (in comparison with the interacting system) is formally the effective one.
        
        moments(0, 0) << 0.0, 0.0,
                         0.0, 0.0;  // Set first moment
        moments(0, 1) << 4.0 * t * t, 0.0,
                         0.0,         4.0 * t * t;  // Set second moment
        moments(1, 0) = moments(0, 0);
        moments(1, 1) = moments(0, 1);
        H0dec->moments(std::move(moments));
        
        H0dec->primVecs((Eigen::Matrix2d() << q, 0, 0, 1).finished());
        
        H0dec->type("dimer_mag_2d");   // general, bethe, bethe_dimer, dimer_mag_2d
        H0dec->computeBands((ArrayXsizet(2) << nkx, nky).finished());
        H0dec->computeDOS(nbins4dos);
    }
    */
    
    
//#ifdef PADE_NOT_USE_MPFR
//    PadeApproximant2XXld pade;
//#else
//    mpfr::mpreal::set_default_prec(mpprec);  // Set default precision for Pade interpolation
//    PadeApproximant2XXmpreal pade;
//#endif
    MQEMContinuator<2, Eigen::Dynamic, Eigen::Dynamic> mqem;
    mqem.parameters.at("simple_default_model") = simple_default_model;
    mqem.parameters.at("secant_max_iter") = secant_maxiter;
    mqem.parameters.at("secant_tol") = secant_tol;
    mqem.parameters.at("secant_damp") = secant_damp;
    mqem.parameters.at("matrix_invertible_threshold") = mat_inv_threshold;
    mqem.parameters.at("Pulay_mixing_param") = pulay_mix;
    mqem.parameters.at("Pulay_history_size") = pulay_histsize;
    mqem.parameters.at("Pulay_period") = pulay_period;
    mqem.parameters.at("Gaussian_sigma") = U / 2.0;
    mqem.parameters.at("Gaussian_shift") = -mu_eff;
    mqem.parameters.at("alpha_max_fac") = alpha_maxfac;
    mqem.parameters.at("alpha_info_fit_fac") = alpha_infofitfac;
    mqem.parameters.at("alpha_init_fraction") = alpha_initfrac;
    mqem.parameters.at("alpha_stop_slope") = alpha_stopslope;
    mqem.parameters.at("alpha_stop_step") = alpha_stopstep;
    mqem.parameters.at("Pulay_tolerance") = pulay_tol;
    mqem.parameters.at("Pulay_max_iteration") = pulay_maxiter;
    mqem.parameters.at("alpha_spec_rel_err") = alpha_dAtol;
    mqem.parameters.at("alpha_step_min_ratio") = alpha_rmin;
    mqem.parameters.at("alpha_step_max_ratio") = alpha_rmax;
    mqem.parameters.at("alpha_step_scale") = alpha_rscale;
    mqem.parameters.at("alpha_capacity") = alpha_capacity;
    mqem.parameters.at("alpha_cache_all") = alpha_cacheall;
    mqem.parameters.at("curvature_fit_method") = curvfit_method;
    mqem.parameters.at("arc_fit_size") = arcfit_size;
    mqem.parameters.at("FDfit_damp") = fdfit_damp;
    mqem.parameters.at("FDfit_tolerance") = fdfit_tol;
    mqem.parameters.at("FDfit_max_iteration") = fdfit_maxiter;
    mqem.parameters.at("principal_int_eps") = pint_eps;
    
    const Eigen::ArrayXd midrealfreqs = mqem.midRealFreqs(midrealfreq_anchors_steps);
    if (prank == 0) std::cout << "Middle real frequency grid size for MQEM is " << midrealfreqs.size() << std::endl;
    
    double conductivities[2] = {0.0, 0.0};
    SqMatArray2XXcd Aw, A0, Akw;
    Eigen::Index id0freq;
    //Eigen::ArrayXcd en_idel;
    // For testing
    SqMatArray2XXcd selfentail(2, nfcut + 1, nsites);
    
    if (proc_control == 1) {  // proc_control == 1 for doing analytic continuation only
        SqMatArray2XXcd selfendyn(2, nfcut + 1, nsites, MPI_COMM_WORLD);
        SqMatArray21Xcd selfenstatic(2, 1, nsites, MPI_COMM_WORLD);
        SqMatArray23Xcd selfenmom(2, 3, nsites, MPI_COMM_WORLD);
        SqMatArray2XXd selfenvar(2, nfcut + 1, nsites, MPI_COMM_WORLD);
        
        if (prank == 0) {
            file_id = H5Fopen("solution.h5", H5F_ACC_RDWR, H5P_DEFAULT);
            
            dset_se_dyn_id = H5Dopen2(file_id, "/SelfEnergy/Dynamical", H5P_DEFAULT);
            dset_se_static_id = H5Dopen2(file_id, "/SelfEnergy/Static", H5P_DEFAULT);
            dset_se_mom_id = H5Dopen2(file_id, "/SelfEnergy/Moments", H5P_DEFAULT);
            dset_se_var_id = H5Dopen2(file_id, "/SelfEnergy/Variance", H5P_DEFAULT);
            
            //loadData("selfenergy_dyn.txt", selfendyn);
            //loadData("selfenergy_var.txt", selfenvar);
            //loadData("selfenergy_static.txt", selfenstatic);
            //loadData("selfenergy_moms.txt", selfenmom);
            status = H5Dread(dset_se_dyn_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, selfendyn().data());
            status = H5Dread(dset_se_static_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, selfenstatic().data());
            status = H5Dread(dset_se_mom_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, selfenmom().data());
            status = H5Dread(dset_se_var_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, selfenvar().data());
            
            status = H5Dclose(dset_se_dyn_id);
            status = H5Dclose(dset_se_static_id);
            status = H5Dclose(dset_se_var_id);
        }
        selfendyn.broadcast(0);
        selfenvar.broadcast(0);
        selfenstatic.broadcast(0);
        selfenmom.broadcast(0);
        
        //pade.build(selfendyn, beta, Eigen::ArrayXi::LinSpaced(ndatalens, mindatalen, maxdatalen),
        //           Eigen::ArrayXi::LinSpaced(nstartfreqs, minstartfreq, maxstartfreq),
        //           Eigen::ArrayXi::LinSpaced(ncoefflens, mincoefflen, maxcoefflen), MPI_COMM_WORLD);
        Eigen::ArrayXd mats_freq = Eigen::ArrayXd::LinSpaced(selfendyn.dim1(), M_PI / beta, (2 * selfendyn.dim1() - 1) * M_PI / beta);
        DMFTIterator::fitSelfEnMoms23(mats_freq, selfendyn, selfenvar, tailstart, selfenmom);  // Refit second and third moments of self-energy
        if (prank == 0) {
            //printData("selfenergy_moms.txt", selfenmom);
            status = H5Dwrite(dset_se_mom_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, selfenmom().data());
            status = H5Dclose(dset_se_mom_id);
            std::cout << "Output selfenergy_moms" << std::endl;
            // For testing
            for (Eigen::Index s = 0; s < 2; ++s) {
                for (Eigen::Index n = 0; n <= nfcut; ++n) {
                    selfentail(s, n) = selfenmom(s, 0) / (mats_freq(n) * 1i)
                    + selfenmom(s, 1) / (-mats_freq(n) * mats_freq(n))
                    + selfenmom(s, 2) / (-1i * mats_freq(n) * mats_freq(n) * mats_freq(n));
                }
            }
            //printData("selfenergy_tail.txt", selfentail);
            dset_se_tail_id = H5Dopen2(file_id, "/SelfEnergy/Tail", H5P_DEFAULT);
            status = H5Dwrite(dset_se_tail_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, selfentail().data());
            status = H5Dclose(dset_se_tail_id);
            std::cout << "Output selfenergy_tail" << std::endl;
        }
        mqem.assembleKernelMatrix(mats_freq, n_lrealfreq, midrealfreqs, n_rrealfreq);
        mqem.realFreqGrid().abs().minCoeff(&id0freq);
        if (prank == 0) {
            //printData("real_freqs.txt", mqem.realFreqGrid());
            dset_id = H5Dopen2(file_id, "/SelfEnergy/RealFreq", H5P_DEFAULT);
            status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mqem.realFreqGrid().data());
            status = H5Dclose(dset_id);
            std::cout << "Output real_freqs; Approximately zero frequency is " << mqem.realFreqGrid()(id0freq) << " at " << id0freq << std::endl;
        }
        mqem.computeSpectra(mats_freq, selfendyn, selfenvar, selfenmom);
        //pade.computeSpectra(selfenstatic, *H0, nenergies, minenergy, maxenergy, delenergy, physonly);
        mqem.computeRetardedFunc(selfenstatic);
        computeSpectraW(*H0, mqem.retardedFunc(), mqem.realFreqGrid(), Aw);
        computeSpectraKW0(*H0, mqem.retardedFunc(), id0freq, A0);
        computeSpectraKW(*H0, mqem.retardedFunc(), mqem.realFreqGrid(), kidpath, Akw);
        if (prank == 0) {
            dset_se_dm_id = H5Dopen2(file_id, "/SelfEnergy/DefaultModel", H5P_DEFAULT);
            dset_se_ret_id = H5Dopen2(file_id, "/SelfEnergy/Retarded", H5P_DEFAULT);
            dset_sp_w_id = H5Dopen2(file_id, "/Spectrum/kIntegrated", H5P_DEFAULT);
            dset_sp_k0_id = H5Dopen2(file_id, "/Spectrum/FermiSurface", H5P_DEFAULT);
            dset_sp_kw_id = H5Dopen2(file_id, "/Spectrum/kFreq", H5P_DEFAULT);
            
            //printData("default_model.txt", mqem.defaultModel());
            status = H5Dwrite(dset_se_dm_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, mqem.defaultModel()().data());
            std::cout << "Output default_model" << std::endl;
            //printData("selfenergy_retarded.txt", mqem.retardedFunc());
            status = H5Dwrite(dset_se_ret_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, mqem.retardedFunc()().data());
            std::cout << "Output selfenergy_retarded" << std::endl;
            //printData("spectraw.txt", Aw);
            status = H5Dwrite(dset_sp_w_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, Aw().data());
            std::cout << "Output spectramatrix" << std::endl;
            //printData("spectrakw0.txt", A0);
            status = H5Dwrite(dset_sp_k0_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, A0().data());
            std::cout << "Output spectra0freq" << std::endl;
            //printData("spectrakw.txt", Akw);
            status = H5Dwrite(dset_sp_kw_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, Akw().data());
            std::cout << "Output spectrakw" << std::endl;
            printData("mqem_diagnosis.txt", mqem.diagnosis(0), std::numeric_limits<double>::max_digits10);
            std::cout << "Output mqem_diagnosis.txt" << std::endl;
            //std::cout << "#spectra: " << pade.nPhysSpectra().sum() << std::endl;
            std::cout << "Optimal log10(alpha) for spin up: " << mqem.optimalLog10alpha(0) << " at " << mqem.optimalAlphaIndex(0) << std::endl;
            
            status = H5Dclose(dset_se_dm_id);
            status = H5Dclose(dset_se_ret_id);
            status = H5Dclose(dset_sp_w_id);
            status = H5Dclose(dset_sp_k0_id);
            status = H5Dclose(dset_sp_kw_id);
        }
        
        //en_idel = mqem.realFreqGrid() + Eigen::ArrayXcd::Constant(mqem.realFreqGrid().size(), 1i * delenergy);
        conductivities[0] = longitConduc(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector(), intalg_);
        if (computesigmaxy) {
            if (q == 1) conductivities[1] = hallConducCoeff(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector(), intalg_);
            else conductivities[1] = hallConduc(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector(), intalg_);
        }
        if (prank == 0) {
            attr_cond_id = H5Aopen(file_id, "/Conductivities", H5P_DEFAULT);
            status = H5Awrite(attr_cond_id, H5T_NATIVE_DOUBLE, conductivities);
            status = H5Aclose(attr_cond_id);
            status = H5Fclose(file_id);
            std::cout << std::scientific << std::setprecision(5) << "sigmaxx = " << conductivities[0] << std::endl;
            if (computesigmaxy) {
                if (q == 1) std::cout << std::scientific << std::setprecision(5) << "sigmaxy / (p / q) (p / q -> 0) = " << conductivities[1] << std::endl;
                else std::cout << std::scientific << std::setprecision(5) << "sigmaxy = " << conductivities[1] << std::endl;
            }
        }
        
        MPI_Finalize();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<60> > duration = stop - start;   // Unit is hours
        if (prank == 0) std::cout << "Execution time: " << duration.count() << " minutes" << std::endl;
        return 0;
    }
    else if (proc_control == 2) {  // proc_control == 2 for only calculating curvature of misfit curve for MQEM
        if (prank == 0) {
            Eigen::ArrayX4d misfit;
            Eigen::Index opt_alpha_ind;
            loadData("mqem_diagnosis.txt", misfit);
            //MQEMContinuator2XX::fitArc(misfit.leftCols<2>(), misfit.col(2), alpha_fitsize);
            std::cout << "Fitted parameters of FD function: " << mqem.fitFDFunc(misfit.leftCols<2>(), misfit.rightCols<2>()).transpose() << std::endl;
            printData("mqem_diagnosis.txt", misfit, std::numeric_limits<double>::max_digits10);
            std::cout << "Output mqem_diagnosis.txt" << std::endl;
            misfit.col(3).maxCoeff(&opt_alpha_ind);
            std::cout << "Optimal log10(alpha) for spin up: " << misfit(opt_alpha_ind, 0) << " at " << opt_alpha_ind << std::endl;
        }
        MPI_Finalize();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<60> > duration = stop - start;   // Unit is hours
        if (prank == 0) std::cout << "Execution time: " << duration.count() << " minutes" << std::endl;
        return 0;
    }
    
    
    if (measurewhat == "S") ntau4eiwt = 0;  // Not allocate eiwt array anyway if measuring S
    auto G0 = std::make_shared<BareGreenFunction>(beta, nsites, nfcut, ntau, ntau4eiwt, MPI_COMM_WORLD);
//    std::complex<double> omega;
//    for (int o = 0; o <= nfcut; ++o) {
//        omega = (2.0 * o + 1.0) * M_PI / beta;
//        G0->fourierCoeffs()(0, o)(0, 0) = -1.0 / (1i * omega + 1i);  // Note this is different from conventional definition
//        G0->fourierCoeffs()(1, o)(0, 0) = -1.0 / (1i * omega + 1i);
//    }
//    G0->fourierInversion();
//    if (prank == 0) {
//        printG("G0.txt", G0);
//    }

    auto G = std::make_shared<GreenFunction>(beta, nsites, nfcut, ntau, nbins4S, MPI_COMM_WORLD);
    
    //MPI_Barrier(MPI_COMM_WORLD);
    //usleep(1000 * prank);
    //std::this_thread::sleep_for(std::chrono::milliseconds(prank));
    //std::cout << "rank " << prank << " of " << psize << ": mastered size = " << G->fourierCoeffs().mastFlatPart().size() << ", mastered start = "
    // << G->fourierCoeffs().mastFlatPart().displ() << std::endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    //sleep(1);   // Wait 1 s
    //std::this_thread::sleep_for(std::chrono::seconds(1));
    //if (prank == 0) std::cout << sep << std::endl;

    //auto impproblem = std::make_shared<ImpurityProblem>(H0, G0, U, K, G);
    //if (loc_corr) impproblem = std::make_shared<ImpurityProblem>(H0dec, G0, U, K, G);
    //else impproblem = std::make_shared<ImpurityProblem>(H0, G0, U, K, G);

    CTAUXImpuritySolver impsolver(H0, G0, U, K, G);
    impsolver.parameters.at("markov chain length") = markovchainlength;
    impsolver.parameters.at("QMC time limit") = qmctimelimit;
    impsolver.parameters.at("#warm up steps") = warmupsize;
    impsolver.parameters.at("measure period") = measureperiod;
    // impsolver.parameters.at("does measure") = false;
    impsolver.parameters.at("measure what") = measurewhat;
    impsolver.parameters.at("histogram max order") = histmaxorder;
    impsolver.parameters.at("magnetic order") = magneticorder;
    impsolver.parameters.at("verbosity") = verbosity;
    
    DMFTIterator dmft(H0, G0, G);
    dmft.parameters.at("G0 update step size") = G0stepsize;
    //dmft.parameters.at("convergence type") = converge_type;
    dmft.parameters.at("convergence criterion") = converge_criterion;
    dmft.parameters.at("local correlation") = loc_corr;
    dmft.parameters.at("high_freq_tail_start") = tailstart;
    
    
    
    std::pair<bool, double> converg;
    int cond_computed_times = 0;
    double interr;
    // double varmax, var;
    auto tstart = std::chrono::high_resolution_clock::now();
    auto tend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<60> > tdur;  // Unit is minutes
//    std::array<Eigen::Index, 2> so;
//    std::complex<double> zeta, zetasq;
//    double sgn;
    
    // Initialize self-energy and Green's functions; Bare Green's function data consists of imaginary time values, Fourier coefficients, and moments (not including
    // imaginary times, Matsubara frequencies, and cubic spline). Therefore, must set these 3 data to set bare Green's function.
    if (ansatz == "insulator") {
        G0->computeMoments(*H0);
        auto G0wmastpart = G0->fourierCoeffs().mastFlatPart();
        std::array<Eigen::Index, 2> so;
        //if (loc_corr)
        //    for (Eigen::Index i = 0; i < G0wmastpart.size(); ++i) {
        //        so = G0wmastpart.global2dIndex(i);
        //        // Hybridization is set to zero to indicate the insulating ansatz (insulating bath should not screen the impurity)
        //        G0wmastpart[i].noalias() = -((1i * G0->matsubFreqs()(so[1]) + mu_eff) * Eigen::MatrixXcd::Identity(nc, nc) - H0dec->moments()(so[0], 0)).inverse();
        //    }
        //else
        for (Eigen::Index i = 0; i < G0wmastpart.size(); ++i) {
            so = G0wmastpart.global2dIndex(i);
            // Hybridization is set to zero to indicate the insulating ansatz (insulating bath should not screen the impurity)
            G0wmastpart[i].noalias() = -((1i * G0->matsubFreqs()(so[1]) + mu_eff) * Eigen::MatrixXcd::Identity(nsites, nsites) - H0->moments()(so[0], 0)).inverse();
        }
        G0wmastpart.allGather();
        G0->invFourierTrans();
    }
    else if (ansatz == "metal") {  // Initialization for metallic solution (zero-U limit)
        dmft.dynSelfEnergy().mastFlatPart()().setZero();
        dmft.staticSelfEnergy()().setZero();
        dmft.updateLatticeGF();
        dmft.updateBathGF();
    }
    else {  // Read in initial G0 from file, can be used to do continuation calculations
        G0->computeMoments(*H0);
        if (prank == 0) {
            std::ifstream fin("G0matsubara.txt");
            if (fin.is_open()) {
                fin >> G0->fourierCoeffs();
                fin.close();
            }
            else std::cout << "Unable to open file" << std::endl;
        }
        G0->fourierCoeffs().broadcast(0);
        G0->invFourierTrans();
    }
//    if (H0->dosType() == "semicircular") {
//        for (Eigen::Index i = 0; i < G->fourierCoeffs().mastPartSize(); ++i) {
//            so = G->fourierCoeffs().index2DinPart(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
//            zeta = 1i * ((2 * so[1] + 1) * M_PI / beta) + H0->mu;
//            zetasq = zeta * zeta + 1e-6i;
//            sgn = zetasq.imag() > 0 ? 1.0 : -1.0;
//            G->fourierCoeffs().masteredPart(i)(0, 0) = -0.5 * (zeta - sgn * std::sqrt(zetasq - 4.0));
//        }
//    }
    
    mqem.assembleKernelMatrix(G->matsubFreqs(), n_lrealfreq, midrealfreqs, n_rrealfreq);
    mqem.realFreqGrid().abs().minCoeff(&id0freq);
    
    // Write data to hdf5 file
    if (prank == 0) {
        file_id = H5Fcreate("solution.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        dims_mat[0] = 2;
        dspace_id = H5Screate_simple(1, dims_mat, NULL);
        attr_cond_id = H5Acreate2(file_id, "/Conductivities", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(dspace_id);
        
        grp_id = H5Gcreate2(file_id, "/BareElectrStruct", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dims_mat[0] = H0->bands().rows(); dims_mat[1] = H0->bands().cols();
        dspace_id = H5Screate_simple(RANK_MAT, dims_mat, NULL);
        dset_id = H5Dcreate2(grp_id, "Bands", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(dspace_id);
        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, H0->bands().data());
        status = H5Dclose(dset_id);
        //printData("bands.txt", H0->bands().transpose());
        std::cout << "Output bands" << std::endl;
        dims_mat[0] = H0->dos().rows(); dims_mat[1] = H0->dos().cols();
        dspace_id = H5Screate_simple(RANK_MAT, dims_mat, NULL);
        dset_id = H5Dcreate2(grp_id, "DOS", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(dspace_id);
        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, H0->dos().data());
        status = H5Dclose(dset_id);
        //printData("dos.txt", H0->dos());
        std::cout << "Output dos" << std::endl;
        status = H5Gclose(grp_id);
        
        dims_gf[0] = 2; dims_gf[1] = ntau; dims_gf[2] = nsites; dims_gf[3] = nsites;
        dspace_t_id = H5Screate_simple(RANK_GF, dims_gf, NULL);
        dims_gf[1] = nfcut + 1;
        dspace_iw_id = H5Screate_simple(RANK_GF, dims_gf, NULL);
    
        grp_g0_id = H5Gcreate2(file_id, "/BathGreenFunc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dset_g0_t_id = H5Dcreate2(grp_g0_id, "ImagTimeDomain", complex_id, dspace_t_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dset_g0_iw_id = H5Dcreate2(grp_g0_id, "MatsFreqDomain", complex_id, dspace_iw_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        grp_gf_id = H5Gcreate2(file_id, "/GreenFunc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dset_gf_t_id = H5Dcreate2(grp_gf_id, "ImagTimeDomain", complex_id, dspace_t_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dset_gf_iw_id = H5Dcreate2(grp_gf_id, "MatsFreqDomain", complex_id, dspace_iw_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
     
        grp_se_id = H5Gcreate2(file_id, "/SelfEnergy", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dset_se_dyn_id = H5Dcreate2(grp_se_id, "Dynamical", complex_id, dspace_iw_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dims_static[0] = 2; dims_static[1] = nsites; dims_static[2] = nsites;
        dspace_id = H5Screate_simple(RANK_STATIC, dims_static, NULL);
        dset_se_static_id = H5Dcreate2(grp_se_id, "Static", complex_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(dspace_id);
        dims_gf[1] = 3;
        dspace_id = H5Screate_simple(RANK_GF, dims_gf, NULL);
        dset_se_mom_id = H5Dcreate2(grp_se_id, "Moments", complex_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(dspace_id);
        dset_se_var_id = H5Dcreate2(grp_se_id, "Variance", H5T_NATIVE_DOUBLE, dspace_iw_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dset_se_tail_id = H5Dcreate2(grp_se_id, "Tail", complex_id, dspace_iw_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dims_gf[1] = mqem.realFreqGrid().size();
        dspace_id = H5Screate_simple(1, &dims_gf[1], NULL);
        dset_id = H5Dcreate2(grp_se_id, "RealFreq", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(dspace_id);
        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mqem.realFreqGrid().data());
        status = H5Dclose(dset_id);
        std::cout << "Output real_freqs; Approximately zero frequency is " << mqem.realFreqGrid()(id0freq) << " at " << id0freq << std::endl;
        dspace_w_id = H5Screate_simple(RANK_GF, dims_gf, NULL);
        dset_se_ret_id = H5Dcreate2(grp_se_id, "Retarded", complex_id, dspace_w_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dset_se_dm_id = H5Dcreate2(grp_se_id, "DefaultModel", complex_id, dspace_w_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        grp_sp_id = H5Gcreate2(file_id, "/Spectrum", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dset_sp_w_id = H5Dcreate2(grp_sp_id, "kIntegrated", complex_id, dspace_w_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dims_gf[1] = H0->kGridSizes().prod();
        dspace_id = H5Screate_simple(RANK_GF, dims_gf, NULL);
        dset_sp_k0_id = H5Dcreate2(grp_sp_id, "FermiSurface", complex_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(dspace_id);
        dims_sp[0] = 2; dims_sp[1] = kidpath.cols(); dims_sp[2] = mqem.realFreqGrid().size(); dims_sp[3] = nsites; dims_sp[4] = nsites;
        dspace_id = H5Screate_simple(RANK_SP, dims_sp, NULL);
        dset_sp_kw_id = H5Dcreate2(grp_sp_id, "kFreq", complex_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(dspace_id);
        
        status = H5Sclose(dspace_t_id);
        status = H5Sclose(dspace_iw_id);
        status = H5Sclose(dspace_w_id);
    }
    // Test
    //if (prank == 0) {
    //    G->valsOnTauGrid()().setOnes();
    //    status = H5Dwrite(dset_gf_t_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, G->valsOnTauGrid()().data());
    //
    //   status = H5Dclose(dset_gf_t_id);
    //    status = H5Dclose(dset_gf_w_id);
    //    status = H5Gclose(grp_gf_id);
    //    status = H5Fclose(file_id);
    //}
    //MPI_Finalize();
    //return 0;
    
    
    //if (prank == 0) {
    //    printData("real_freqs.txt", mqem.realFreqGrid());
    //    std::cout << "Output real_freqs.txt; Approximately zero frequency is " << mqem.realFreqGrid()(id0freq) << " at " << id0freq << std::endl;
    //}
    
    bool computesigma;
    double density, density_old = 1.0, dmu = mu_eff;  //  mueff_old = 0.0;  // Initialize to half filling for fixing density
    
    const int cw = 13;
    std::string dash(cw, '-');
    std::ofstream fiter;
    if (prank == 0) {
        fiter.open("iterations.txt", std::fstream::out | std::fstream::trunc);
        // std::cout << "\u03A3 \u03C9" << std::endl;
        if (measurewhat == "S") {
            fiter << std::setw(cw / 2 + 1) << " iter" << std::setw(cw + 1) << " converg" << std::setw(cw + 1) << " <order>" << std::setw(cw + 1)
            << " Im<S0>/w0" << std::setw(2 * cw + 2) << " var<n> / <n>" << std::setw(cw + 1) << " <Sz0*Sz1>" << std::setw(cw + 1) << " <sign>"
            << std::setw(cw + 1) << " <n> int err" << std::setw(cw + 1) << " sxx (e^2/h)";
            if (computesigmaxy) {
                if (q == 1) fiter << std::setw(cw + 1) << " sxy/(p/q)";
                else fiter << std::setw(cw + 1) << " sxy (e^2/h)";
            }
            fiter << std::endl;
            fiter << " "  << std::string(cw / 2, '-') << " "               << dash       << " "               << dash       << " "
            << dash         << " "      << std::string(2 * cw + 1, '-') << " "               << dash         << " "               << dash
            << " "               << dash           << " "               << dash;
            if (computesigmaxy) fiter << " "               << dash;
            fiter << std::endl;
        }
        else if (measurewhat == "G") {
            fiter << std::setw(cw / 2 + 1) << " iter" << std::setw(cw + 1) << " converg" << std::setw(cw + 1) << " <order>" << std::setw(cw + 1)
            << " Im<S0>/w0" << std::setw(2 * cw + 2) << " var<n> / <n>" << std::setw(cw + 1) << " <Sz0*Sz1>" << std::setw(cw + 1) << " <sign>"
            << std::setw(cw + 1) << " sxx (e^2/h)";
            if (computesigmaxy) {
                if (q == 1) fiter << std::setw(cw + 1) << " sxy/(p/q)";
                else fiter << std::setw(cw + 1) << " sxy (e^2/h)";
            }
            fiter << std::endl;
            fiter << " "  << std::string(cw / 2, '-') << " "               << dash       << " "               << dash       << " "
            << dash         << " "      << std::string(2 * cw + 1, '-') << " "               << dash         << " "               << dash
            << " "               << dash;
            if (computesigmaxy) fiter << " "               << dash;
            fiter << std::endl;
        }
        fiter << std::scientific << std::setprecision(5);
    }
    do {
        dmft.incrementIter();
        
        if (prank == 0) {
            std::cout << "DMFT iteration " << dmft.numIterations() << ":" << std::endl;
            //printData("G0.txt", G0->valsOnTauGrid());
            status = H5Dwrite(dset_g0_t_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, G0->valsOnTauGrid()().data());
            std::cout << "    Output G0" << std::endl;
            //printData("G0matsubara.txt", G0->fourierCoeffs());
            status = H5Dwrite(dset_g0_iw_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, G0->fourierCoeffs()().data());
            std::cout << "    Output G0matsubara" << std::endl;
        }
        
        if (prank == 0) std::cout << "    Impurity solver starts solving..." << std::endl;
        tstart = std::chrono::high_resolution_clock::now();
        interr = impsolver.solve();
        tend = std::chrono::high_resolution_clock::now();
        tdur = tend - tstart;
        if (prank == 0) {  // Output obtained result ASAP
            printData("histogram.txt", impsolver.vertexOrderHistogram());
            std::cout << "    Output histogram.txt" << std::endl;
            //printData("G.txt", G->valsOnTauGrid());
            status = H5Dwrite(dset_gf_t_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, G->valsOnTauGrid()().data());
            std::cout << "    Output G" << std::endl;
            //printData("Gmatsubara.txt", G->fourierCoeffs());
            status = H5Dwrite(dset_gf_iw_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, G->fourierCoeffs()().data());
            std::cout << "    Output Gmatsubara" << std::endl;
            std::cout << "    Impurity solver completed solving in " << tdur.count() << " minutes" << std::endl;
        }
        
        dmft.approxSelfEnergy();
        if (prank == 0) {  // Output obtained result ASAP
            //printData("selfenergy_dyn.txt", dmft.dynSelfEnergy(), std::numeric_limits<double>::max_digits10);
            status = H5Dwrite(dset_se_dyn_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, dmft.dynSelfEnergy()().data());
            std::cout << "    Output selfenergy_dyn" << std::endl;
            //printData("selfenergy_static.txt", dmft.staticSelfEnergy(), std::numeric_limits<double>::max_digits10);
            status = H5Dwrite(dset_se_static_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, dmft.staticSelfEnergy()().data());
            std::cout << "    Output selfenergy_static" << std::endl;
            //printData("selfenergy_moms.txt", dmft.selfEnergyMoms(), std::numeric_limits<double>::max_digits10);
            status = H5Dwrite(dset_se_mom_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, dmft.selfEnergyMoms()().data());
            std::cout << "    Output selfenergy_moms" << std::endl;
            //printData("selfenergy_var.txt", dmft.selfEnergyVar(), std::numeric_limits<double>::max_digits10);
            status = H5Dwrite(dset_se_var_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dmft.selfEnergyVar()().data());
            std::cout << "    Output selfenergy_var" << std::endl;
            // For testing
            for (Eigen::Index s = 0; s < 2; ++s) {
                for (Eigen::Index n = 0; n <= nfcut; ++n) {
                    selfentail(s, n) = dmft.selfEnergyMoms()(s, 0) / (G0->matsubFreqs()(n) * 1i)
                    + dmft.selfEnergyMoms()(s, 1) / (-G0->matsubFreqs()(n) * G0->matsubFreqs()(n))
                    + dmft.selfEnergyMoms()(s, 2) / (-1i * G0->matsubFreqs()(n) * G0->matsubFreqs()(n) * G0->matsubFreqs()(n));
                }
            }
            //printData("selfenergy_tail.txt", selfentail);
            status = H5Dwrite(dset_se_tail_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, selfentail().data());
            std::cout << "    Output selfenergy_tail" << std::endl;
        }
        
        dmft.updateLatticeGF();
        
        density = G->densities().sum() / nsites;  // Electron density per site
        converg = dmft.checkConvergence();  // Only use G and Glat at the same step
        computesigma = converg.first;
        if (density_goal >= 0.0) computesigma = computesigma && std::abs(density - density_goal) < density_error;
        computesigma = computesigma || dmft.numIterations() == nitmax;
        
        // Calculate conductivities
        if (computesigma) {
            if (prank == 0) std::cout << "    MQEM starts analytic continuation of self-energy..." << std::endl;
            tstart = std::chrono::high_resolution_clock::now();
            //dmft.selfEnergy().mastFlatPart().allGather();
            //pade.build(dmft.dynSelfEnergy(), beta, Eigen::ArrayXi::LinSpaced(ndatalens, mindatalen, maxdatalen),
            //           Eigen::ArrayXi::LinSpaced(nstartfreqs, minstartfreq, maxstartfreq),
            //           Eigen::ArrayXi::LinSpaced(ncoefflens, mincoefflen, maxcoefflen), MPI_COMM_WORLD);
            //pade.computeSpectra(dmft.staticSelfEnergy(), *H0, nenergies, minenergy, maxenergy, delenergy, physonly);
            mqem.computeSpectra(G->matsubFreqs(), dmft.dynSelfEnergy(), dmft.selfEnergyVar(), dmft.selfEnergyMoms());
            mqem.computeRetardedFunc(dmft.staticSelfEnergy());
            computeSpectraW(*H0, mqem.retardedFunc(), mqem.realFreqGrid(), Aw);
            computeSpectraKW0(*H0, mqem.retardedFunc(), id0freq, A0);
            computeSpectraKW(*H0, mqem.retardedFunc(), mqem.realFreqGrid(), kidpath, Akw);
            tend = std::chrono::high_resolution_clock::now();
            tdur = tend - tstart;
            if (prank == 0) {  // Output obtained result ASAP
                //printData("default_model.txt", mqem.defaultModel());
                status = H5Dwrite(dset_se_dm_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, mqem.defaultModel()().data());
                std::cout << "    Output default_model" << std::endl;
                //printData("selfenergy_retarded.txt", mqem.retardedFunc());
                status = H5Dwrite(dset_se_ret_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, mqem.retardedFunc()().data());
                std::cout << "    Output selfenergy_retarded" << std::endl;
                //printData("spectraw.txt", Aw);
                status = H5Dwrite(dset_sp_w_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, Aw().data());
                std::cout << "    Output spectraw" << std::endl;
                //printData("spectrakw0.txt", A0);
                status = H5Dwrite(dset_sp_k0_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, A0().data());
                std::cout << "    Output spectrakw0" << std::endl;
                //printData("spectrakw.txt", Akw);
                status = H5Dwrite(dset_sp_kw_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, Akw().data());
                std::cout << "    Output spectrakw" << std::endl;
                printData("mqem_diagnosis.txt", mqem.diagnosis(0), std::numeric_limits<double>::max_digits10);
                std::cout << "    Output mqem_diagnosis.txt" << std::endl;
                std::cout << "    MQEM completed analytic continuation in " << tdur.count() << " minutes" << std::endl;
                std::cout << "    Optimal log10(alpha) for spin up: " << mqem.optimalLog10alpha(0) << " at " << mqem.optimalAlphaIndex(0) << std::endl;
                //std::cout << "    #spectra = " << pade.nPhysSpectra()(0) << " (up), " << pade.nPhysSpectra()(1) << " (down)" << std::endl;
            }
            
            if (prank == 0) std::cout << "    Start computing conductivities..." << std::endl;
            tstart = std::chrono::high_resolution_clock::now();
            conductivities[0] = longitConduc(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector(), intalg_);
            if (computesigmaxy) {
                if (q == 1) conductivities[1] = hallConducCoeff(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector(), intalg_);
                else conductivities[1] = hallConduc(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector(), intalg_);
            }
            tend = std::chrono::high_resolution_clock::now();
            tdur = tend - tstart;
            if (prank == 0) std::cout << "    Computed conductivities in " << tdur.count() << " minutes" << std::endl;
            ++cond_computed_times;
        }
        
        if (prank == 0) {
            if (measurewhat == "S") {
                fiter << " " << std::setw(cw / 2) << dmft.numIterations() << " " << std::setw(cw) << converg.second << " " << std::setw(cw)
                << impsolver.aveVertexOrder() << " " << std::setw(cw) << std::imag(dmft.dynSelfEnergy()(0, 0, 0, 0) + dmft.staticSelfEnergy()(0, 0, 0, 0)) / (M_PI / beta)
                << " " << std::setw(cw) << G->densStdDev().sum() / nsites << " " << std::setw(cw) << density
                << " " << std::setw(cw) << G->spinCorrelation << " " << std::setw(cw) << impsolver.fermiSign() << " " << std::setw(cw) << interr;
            }
            else if (measurewhat == "G") {
                fiter << " " << std::setw(cw / 2) << dmft.numIterations() << " " << std::setw(cw) << converg.second << " " << std::setw(cw)
                << impsolver.aveVertexOrder() << " " << std::setw(cw) << std::imag(dmft.dynSelfEnergy()(0, 0, 0, 0) + dmft.staticSelfEnergy()(0, 0, 0, 0)) / (M_PI / beta)
                << " " << std::setw(cw) << G->densStdDev().sum() / nsites << " " << std::setw(cw) << density
                << " " << std::setw(cw) << G->spinCorrelation << " " << std::setw(cw) << impsolver.fermiSign();
            }
            if (computesigma) {
                status = H5Awrite(attr_cond_id, H5T_NATIVE_DOUBLE, conductivities);
                fiter << " " << std::setw(cw) << conductivities[0];
                if (computesigmaxy) fiter << " " << std::setw(cw) << conductivities[1];
            }
            else {
                fiter << " " << std::setw(cw) << "--";
                if (computesigmaxy) fiter << " " << std::setw(cw) << "--";
            }
            fiter << std::endl;
        }
        
        if (cond_computed_times == n_computecond) break;
        else if (dmft.numIterations() == nitmax) {
            if (prank == 0) {
                fiter << sep << std::endl;
                fiter << ">>>>>> DMFT self-consistency iteration did not converge! <<<<<<" << std::endl;
            }
            break;
        }
        
        if (density_goal >= 0.0 && converg.first) {
            // Secant iteration for finding root of n(mu_eff) - n_goal = 0
            dmu *= -(density - density_goal) / (density - density_old);
            //H0->chemPot(mu_eff - (mu_eff - mueff_old) / (density - density_old) * (density - density_goal));
            //mueff_old = mu_eff;
            H0->chemPot(H0->chemPot() + dmu);
            density_old = density;
            //mu_eff = H0->chemPot();
            mqem.parameters.at("Gaussian_shift") = -H0->chemPot();
            if (prank == 0) std::cout << "Adjusted effective chemical potential by " << dmu << " to " << H0->chemPot() << std::endl;
        }
        
        dmft.updateBathGF();
    } while (true);  // Main iteration stops here
    
    if (prank == 0) {
        fiter.close();
        
        status = H5Aclose(attr_cond_id);
        
        status = H5Dclose(dset_g0_t_id);
        status = H5Dclose(dset_g0_iw_id);
        status = H5Dclose(dset_gf_t_id);
        status = H5Dclose(dset_gf_iw_id);
        status = H5Dclose(dset_se_dyn_id);
        status = H5Dclose(dset_se_static_id);
        status = H5Dclose(dset_se_mom_id);
        status = H5Dclose(dset_se_var_id);
        status = H5Dclose(dset_se_tail_id);
        status = H5Dclose(dset_se_ret_id);
        status = H5Dclose(dset_se_dm_id);
        status = H5Dclose(dset_sp_w_id);
        status = H5Dclose(dset_sp_k0_id);
        status = H5Dclose(dset_sp_kw_id);
        
        status = H5Gclose(grp_g0_id);
        status = H5Gclose(grp_gf_id);
        status = H5Gclose(grp_se_id);
        status = H5Gclose(grp_sp_id);
        
        status = H5Fclose(file_id);
        
        std::cout << sep << std::endl;
    }
    
//    if (prank == 0) {
//        printHistogram("screen", impsolver.vertexOrderHistogram());
//        std::cout << sep << std::endl;
//    }
    
    
    MPI_Finalize();
    
    // Get ending timepoint
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<3600> > duration = stop - start;   // Unit is hours
    if (prank == 0) std::cout << "Execution time: " << duration.count() << " hours" << std::endl;
    
    return 0;
}
