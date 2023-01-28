//
//  main.cpp
//  dmft-renew
//
//  Created by Yin Shi on 4/22/22.
//

#include <iostream>
#include <iomanip>      // std::setw
#include <fstream>
#include <thread>       // std::this_thread::sleep_for
#include <chrono>
#include <limits>
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

// For testing
template <typename ScalarT, int n0, int n1, int nm>
void printData(const std::string& fname, const SqMatArray<ScalarT, n0, n1, nm>& data, const int precision = 6) {
    std::ofstream myfile(fname, std::fstream::out | std::fstream::trunc);
    myfile << std::setprecision(precision);
    if (myfile.is_open()) {
        myfile << data;
        myfile.close();
    }
    else std::cout << "Unable to open file" << std::endl;
}

template <typename Derived>
void printData(const std::string& fname, const Eigen::DenseBase<Derived>& data, const int precision = 6) {
    std::ofstream myfile(fname, std::fstream::out | std::fstream::trunc);
    myfile << std::setprecision(precision);
    if (myfile.is_open()) {
        myfile << data;
        myfile.close();
    }
    else std::cout << "Unable to open file" << std::endl;
}

void printHistogram(const std::string &fname, const ArrayXsizet &vohistogram) {
    if (fname == "screen") {
        std::cout << std::setw(15) << "Vertex order" << std::setw(15) << "Count" << std::endl;
        std::cout << std::setw(15) << "------------" << std::setw(15) << "-----" << std::endl;
        for (int i = 0; i < vohistogram.size(); ++i) {
            std::cout << std::setw(15) << i << std::setw(15) << vohistogram[i] << std::endl;
        }
    }
    else {
        std::ofstream myfile(fname, std::fstream::out | std::fstream::trunc);
        if (myfile.is_open()) {
            myfile << std::setw(15) << "Vertex order" << std::setw(15) << "Count" << std::endl;
            myfile << std::setw(15) << "------------" << std::setw(15) << "-----" << std::endl;
            for (int i = 0; i < vohistogram.size(); ++i) {
                myfile << std::setw(15) << i << std::setw(15) << vohistogram[i] << std::endl;
            }
            myfile.close();
        }
        else std::cout << "Unable to open file" << std::endl;
    }
}

// Derive from the BareHamiltonian class the user-defined Hamiltonian
class Dimer2DinMag : public BareHamiltonian {
public:
    int p, q;  // Constitutes the magnetic field
    
    void constructHamiltonian(const Eigen::VectorXd& k, Eigen::MatrixXcd& H) const override {  // Only need to construct the lower triangular part
        if (q < 1) throw std::range_error("q cannot be less than 1!");
        
        H = Eigen::MatrixXcd::Zero(2 * q, 2 * q);
        int l;
        for (l = 0; l < q; ++l) {
            H(l, l) = 2.0 * m_t(0) * cos(k[1] + (2.0 * M_PI * p) / q * l);   // t array is a protected member of BareHamiltonian
            H(l + q, l + q) = H(l, l);
            H(l + q, l) = m_t(1);
        }
        for (l = 0; l < q - 1; ++l) {
            H(l + 1, l) = m_t(0);
            H(l + q + 1, l + q) = H(l + 1, l);
        }
        const std::complex<double> hop = m_t(0) * std::exp(1i * std::fmod(k[0] * q, 2 * M_PI));
        H(q - 1, 0) += hop;  // "+=" works for all q > 0
        H(0, q - 1) += std::conj(hop);  // This only plays a role in q = 1 case because we only use the lower triangular part of H
        H(2 * q - 1, q) = H(q - 1, 0);
    }
    
    // Only need to construct the lower triangular part
    void constructFermiVelocities(const int coord, const Eigen::VectorXd& k, Eigen::MatrixXcd& v) const override {
        if (coord < 0 || coord > 1) throw std::range_error("Coordinate can only be 0 or 1 for 2D dimer Hubbard model in magnetic fields!");
        if (q < 1) throw std::range_error("q cannot be less than 1!");
        
        v = Eigen::MatrixXcd::Zero(2 * q, 2 * q);
        int l;
        // Note the Fermi velocity matrix is already block diagonal; the spaces of the upper and lower lattices are decoupled.
        if (coord == 0) {  // Fermi velocity in x direction, along which the magnetic unit cell expands
            for (l = 0; l < q - 1; ++l) {
                v(l + 1, l) = -1i * m_t(0);
                v(l + q + 1, l + q) = v(l + 1, l);
            }
            const std::complex<double> hop = m_t(0) * std::exp(1i * std::fmod(k[0] * q, 2 * M_PI));
            const std::complex<double> v0 = 1i * static_cast<double>(q) * hop - 1i * static_cast<double>(q - 1) * hop;
            v(q - 1, 0) += v0;  // "+=" works for all q > 0
            v(0, q - 1) += std::conj(v0);  // This only plays a role in q = 1 case because we only use the lower triangular part of H
            v(2 * q - 1, q) = v(q - 1, 0);
        }
        else if (coord == 1) {  // Fermi velocity in y direction
            for (l = 0; l < q; ++l) {
                v(l, l) = -2.0 * m_t(0) * sin(k[1] + (2.0 * M_PI * p) / q * l);   // t array is a protected member of BareHamiltonian
                v(l + q, l + q) = v(l, l);
            }
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









int main(int argc, char * argv[]) {
    // Get starting timepoint
    auto start = std::chrono::high_resolution_clock::now();
    
    int psize, prank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    
    /*
    // For test
    //Eigen::RowVector4i aa;
    //aa << 3, 0, 1, 2;
    //Eigen::PermutationMatrix<4> perm;
    //perm.indices() = {1, 2, 3, 0};
    //std::cout << aa * perm << std::endl;
    
    std::size_t Nul = 5, Nur = 5, Nw = 51, Niw = 71;
    double omegal = -3.0, omegar = 3.0, invT = 10.0;
    Eigen::ArrayXd iomegas = Eigen::ArrayXd::LinSpaced(Niw, M_PI / invT, (2 * Niw - 1) * M_PI / invT);
    MQEMContinuator<1, Eigen::Dynamic, 2> testmqem;
    //mqem.parameters.at("Gaussian_sigma") = 4.0;
    testmqem.assembleKernelMatrix(iomegas, Nul, omegal, Nw, omegar, Nur);
    
    SqMatArray<std::complex<double>, 1, Eigen::Dynamic, 2> spec(1, testmqem.realFreqGrid().size(), 2), func_mats(1, Niw, 2, MPI_COMM_WORLD);
    Eigen::Matrix2cd rot;
    double theta;
    // Assign model spectral function
    for (std::size_t i = 0; i < testmqem.realFreqGrid().size(); ++i) {
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
    for (std::size_t i = 0; i < Niw; ++i) {
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
    
    MPI_Finalize();
    return 0;
    */
    
    std::string sep("----------------------------------------------------------------------");
    
    // std::random_device rd;
    // std::cout << rd() << std::endl;
    
    std::size_t nc = 2;
    double t = -1.0;
    double tz = -1.0;
    int q = 2;
    int p = 1;
    double U = 4.0;
    double beta = 10.0;
    // mu_eff is the effective chemical potential of the interacting system for the impurity solver while mu = mu_eff + U / 2 is
    // the true chemical potential. mu_eff = 0 corresponds to half-filling, which is default.
    double mu_eff = 0.0;
    double K = 1.0;
    
    std::size_t nkx = 101;
    std::size_t nky = 101;
    std::size_t nbins4dos = 1001;

    std::size_t nfcut = 1000;   // 1000
    std::size_t ntau = 301;  // 1001
    std::size_t ntau4eiwt = 0;
    std::size_t nbins4S = 10001;   // 10001

    std::size_t markovchainlength = 100000000;    // 20000000
    double qmctimelimit = 8.0;  // Unit is minute
    std::size_t measureperiod = 200;    // 200
    std::size_t warmupsize = 10000;
    std::string measurewhat("S");
    std::size_t histmaxorder = 100;
    std::string magneticorder("paramagnetic");
    std::string verbosity("off");
    
    std::string ansatz("metal");
    int nitmax = 20;
    double G0stepsize = 1.0;
    std::string converge_type("Gimp_Glat_max_error");
    double converge_criterion = 0.005;
    
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
    std::size_t nenergies = 201;
    double minenergy = -10.0;
    double maxenergy = 10.0;
    double delenergy = 0.01;
    int mpprec = 256;
     */
    
    // For MQEM analytic continuation
    std::size_t n_lrealfreq = 10;
    double lrealfreq = -5.0;
    std::size_t n_mrealfreq = 51;
    double rrealfreq = 5.0;
    std::size_t n_rrealfreq = 10;
    double pulay_mix = 0.01;
    std::size_t pulay_histsize = 5;
    std::size_t pulay_period = 3;
    double pulay_tol = 1e-6;
    std::size_t pulay_maxiter = 500;
    double gaussian_sig = 1.0;
    double alpha_maxfac = 100.0;
    double alpha_minfac = 1e-5;
    double alpha_stopslope = 0.01;
    double alpha_stopstep = 1e-5;
    double alpha_dAtol = 0.1;
    double alpha_rmin = 0.7;
    double alpha_rmax = 2.0;
    double alpha_rscale = 0.8;
    
    bool analcontrun = false;
    bool computesigmaxy = true;
    bool computecondonce = true;
    
    bool loc_corr = false;   // Whether to use local correlation approximation (off-diagonal elements of self-energy are zero)


    pugi::xml_document doc;
    pugi::xml_node docroot;
    if (!doc.load_file("input.xml")) {
        std::cout << "Error: Cannot load file input.xml" << std::endl;
        return -1;
    }
    docroot = doc.child("input");

    readxml_bcast(nc, docroot, "physical/numSites", MPI_COMM_WORLD, prank);
    readxml_bcast(t, docroot, "physical/hoppingXy", MPI_COMM_WORLD, prank);
    readxml_bcast(tz, docroot, "physical/hoppingXy.hoppingZ", MPI_COMM_WORLD, prank);
    readxml_bcast(q, docroot, "physical/q", MPI_COMM_WORLD, prank);
    readxml_bcast(p, docroot, "physical/q.p", MPI_COMM_WORLD, prank);
    readxml_bcast(beta, docroot, "physical/inverseTemperature", MPI_COMM_WORLD, prank);
    readxml_bcast(U, docroot, "physical/onsiteCoulombRepulsion", MPI_COMM_WORLD, prank);
    readxml_bcast(mu_eff, docroot, "physical/effectiveChemicalPotential", MPI_COMM_WORLD, prank);
    readxml_bcast(nbins4dos, docroot, "numerical/bareDOS/numBins", MPI_COMM_WORLD, prank);
    readxml_bcast(nkx, docroot, "numerical/bareDOS/numBins.numkx", MPI_COMM_WORLD, prank);
    readxml_bcast(nky, docroot, "numerical/bareDOS/numBins.numky", MPI_COMM_WORLD, prank);
    readxml_bcast(nfcut, docroot, "numerical/GreenFunction/frequencyCutoff", MPI_COMM_WORLD, prank);
    readxml_bcast(ntau, docroot, "numerical/GreenFunction/tauGridSize", MPI_COMM_WORLD, prank);
    readxml_bcast(ntau4eiwt, docroot, "numerical/GreenFunction/tauGridSizeOfExpiwt", MPI_COMM_WORLD, prank);
    readxml_bcast(nbins4S, docroot, "numerical/GreenFunction/numTauBinsForSelfEnergy", MPI_COMM_WORLD, prank);
    readxml_bcast(markovchainlength, docroot, "numerical/QMC/MarkovChainLength", MPI_COMM_WORLD, prank);
    readxml_bcast(qmctimelimit, docroot, "numerical/QMC/timeLimit", MPI_COMM_WORLD, prank);
    readxml_bcast(measureperiod, docroot, "numerical/QMC/measurePeriod", MPI_COMM_WORLD, prank);
    readxml_bcast(warmupsize, docroot, "numerical/QMC/numWarmupSteps", MPI_COMM_WORLD, prank);
    readxml_bcast(measurewhat, docroot, "numerical/QMC/whatToMeasure", MPI_COMM_WORLD, prank);
    readxml_bcast(histmaxorder, docroot, "numerical/QMC/maxOrderForHistogram", MPI_COMM_WORLD, prank);
    readxml_bcast(magneticorder, docroot, "numerical/QMC/magneticOrder", MPI_COMM_WORLD, prank);
    readxml_bcast(verbosity, docroot, "numerical/QMC/verbosity", MPI_COMM_WORLD, prank);
    readxml_bcast(ansatz, docroot, "numerical/selfConsistency/ansatz", MPI_COMM_WORLD, prank);
    readxml_bcast(nitmax, docroot, "numerical/selfConsistency/maxIteration", MPI_COMM_WORLD, prank);
    readxml_bcast(G0stepsize, docroot, "numerical/selfConsistency/stepSizeForUpdateG0", MPI_COMM_WORLD, prank);
    readxml_bcast(converge_type, docroot, "numerical/selfConsistency/convergeType", MPI_COMM_WORLD, prank);
    readxml_bcast(converge_criterion, docroot, "numerical/selfConsistency/convergeCriterion", MPI_COMM_WORLD, prank);
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
    readxml_bcast(n_lrealfreq, docroot, "numerical/MQEM/numLeftRealFreqencies", MPI_COMM_WORLD, prank);
    readxml_bcast(lrealfreq, docroot, "numerical/MQEM/leftRealFreqencyAnchor", MPI_COMM_WORLD, prank);
    readxml_bcast(n_mrealfreq, docroot, "numerical/MQEM/numMidRealFreqencies", MPI_COMM_WORLD, prank);
    readxml_bcast(rrealfreq, docroot, "numerical/MQEM/rightRealFreqencyAnchor", MPI_COMM_WORLD, prank);
    readxml_bcast(n_rrealfreq, docroot, "numerical/MQEM/numRightRealFreqencies", MPI_COMM_WORLD, prank);
    readxml_bcast(pulay_mix, docroot, "numerical/MQEM/PulayMixingParameter", MPI_COMM_WORLD, prank);
    readxml_bcast(pulay_histsize, docroot, "numerical/MQEM/PulayHistorySize", MPI_COMM_WORLD, prank);
    readxml_bcast(pulay_period, docroot, "numerical/MQEM/PulayPeriod", MPI_COMM_WORLD, prank);
    readxml_bcast(pulay_tol, docroot, "numerical/MQEM/PulayTolerance", MPI_COMM_WORLD, prank);
    readxml_bcast(pulay_maxiter, docroot, "numerical/MQEM/PulayMaxIteration", MPI_COMM_WORLD, prank);
    readxml_bcast(gaussian_sig, docroot, "numerical/MQEM/GaussianSigma", MPI_COMM_WORLD, prank);
    readxml_bcast(alpha_maxfac, docroot, "numerical/MQEM/alphaMaxFactor", MPI_COMM_WORLD, prank);
    readxml_bcast(alpha_minfac, docroot, "numerical/MQEM/alphaMinFactor", MPI_COMM_WORLD, prank);
    readxml_bcast(alpha_stopslope, docroot, "numerical/MQEM/alphaStopSlope", MPI_COMM_WORLD, prank);
    readxml_bcast(alpha_stopstep, docroot, "numerical/MQEM/alphaStopStep", MPI_COMM_WORLD, prank);
    readxml_bcast(alpha_dAtol, docroot, "numerical/MQEM/alphaSpectrumRelativeError", MPI_COMM_WORLD, prank);
    readxml_bcast(alpha_rmin, docroot, "numerical/MQEM/alphaStepMinRatio", MPI_COMM_WORLD, prank);
    readxml_bcast(alpha_rmax, docroot, "numerical/MQEM/alphaStepMaxRatio", MPI_COMM_WORLD, prank);
    readxml_bcast(alpha_rscale, docroot, "numerical/MQEM/alphaStepScale", MPI_COMM_WORLD, prank);
    readxml_bcast(analcontrun, docroot, "processControl/analyticContinuationOnly", MPI_COMM_WORLD, prank);
    readxml_bcast(computesigmaxy, docroot, "processControl/computeHallConductivity", MPI_COMM_WORLD, prank);
    readxml_bcast(computecondonce, docroot, "processControl/computeConductivityOnce", MPI_COMM_WORLD, prank);
    readxml_bcast(loc_corr, docroot, "processControl/localCorrelation", MPI_COMM_WORLD, prank);

    if (prank == 0) std::cout << sep << std::endl;

    
    // Setup bare Hamiltonian
    auto H0 = std::make_shared<Dimer2DinMag>();
    H0->setMPIcomm(MPI_COMM_WORLD);
    
    H0->hopMatElem(Eigen::Array2cd(t, tz));
    H0->q = q;
    H0->p = p;
    
    H0->chemPot(mu_eff);   // Chemical potential of the noninterating system (in comparison with the interacting system) is formally the effective one.
    
    // Note this constructor: if nc is of type int, on some computers it could be implicitely converted to MPI_Comm type, not std::size_t type,
    // and call another constructor, very dangerous
    SqMatArray22Xcd moments(2, 2, nc);
    moments(0, 0) << 0.0, tz,
                     tz, 0.0;  // Set first moment
    moments(0, 1) << 4.0 * t * t + tz * tz, 0.0,
                     0.0,                   4.0 * t * t + tz * tz;  // Set second moment
    moments(1, 0) = moments(0, 0);
    moments(1, 1) = moments(0, 1);
    H0->moments(moments);
//    H0->firstMoment[0] = Eigen::MatrixXcd::Zero(1, 1);
//    H0->secondMoment[0] = Eigen::MatrixXcd::Ones(1, 1);
//    H0->secondMoment[0] << t * t + tz * tz, 0.0,
//                           0.0, t * t + tz * tz;
    
    H0->primVecs((Eigen::Matrix2d() << q, 0, 0, 1).finished());
    
    H0->type("dimer_mag_2d");   // general, bethe, bethe_dimer, dimer_mag_2d
    H0->computeBands((ArrayXsizet(2) << nkx, nky).finished());
    H0->computeDOS(nbins4dos);
    
//    std::array<double, 2> erange = {-2.0 * std::fabs(t), 2.0 * std::fabs(t)};
//    Eigen::ArrayXd semicircle(nbins4dos);
//    double energy;
//    for (std::size_t ie = 0; ie < semicircle.size(); ++ie) {
//        energy = (ie + 0.5) * (erange[1] - erange[0]) / semicircle.size() + erange[0];
//        semicircle[ie] = sqrt(4.0 * t * t - energy * energy) / (2 * M_PI * t * t);
//    }
    // H0->setDOS(erange, semicircle);
    // H0->setDOS(erange, semicircle);
//    H0->setDOS(erange, semicircle);
    
    Eigen::ArrayXXd bds;
    H0->bands(bds);
    if (prank == 0) {
        printData("bands.txt", bds.transpose());
        printData("dos.txt", H0->dos());
    }
    
    // Test
    // MPI_Finalize();
    // return 0;
    
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
    
    
//#ifdef PADE_NOT_USE_MPFR
//    PadeApproximant2XXld pade;
//#else
//    mpfr::mpreal::set_default_prec(mpprec);  // Set default precision for Pade interpolation
//    PadeApproximant2XXmpreal pade;
//#endif
    MQEMContinuator<2, Eigen::Dynamic, Eigen::Dynamic> mqem;
    mqem.parameters.at("Pulay_mixing_param") = pulay_mix;
    mqem.parameters.at("Pulay_history_size") = pulay_histsize;
    mqem.parameters.at("Pulay_period") = pulay_period;
    mqem.parameters.at("Gaussian_sigma") = gaussian_sig;
    mqem.parameters.at("alpha_max_fac") = alpha_maxfac;
    mqem.parameters.at("alpha_min_fac") = alpha_minfac;
    mqem.parameters.at("alpha_stop_slope") = alpha_stopslope;
    mqem.parameters.at("alpha_stop_step") = alpha_stopstep;
    mqem.parameters.at("Pulay_tolerance") = pulay_tol;
    mqem.parameters.at("Pulay_max_iteration") = pulay_maxiter;
    mqem.parameters.at("alpha_spec_rel_err") = alpha_dAtol;
    mqem.parameters.at("alpha_step_min_ratio") = alpha_rmin;
    mqem.parameters.at("alpha_step_max_ratio") = alpha_rmax;
    mqem.parameters.at("alpha_step_scale") = alpha_rscale;
    
    double sigmaxx = 0.0, sigmaxy = 0.0;
    SqMatArray2XXcd spectra;
    auto spectramastpart = spectra.mastFlatPart();
    //Eigen::ArrayXcd en_idel;
    
    if (analcontrun) {
        SqMatArray2XXcd selfendyn(2, nfcut + 1, nc, MPI_COMM_WORLD);
        SqMatArray2XXd selfenvar(2, nfcut + 1, nc, MPI_COMM_WORLD);
        SqMatArray21Xcd selfenstatic(2, 1, nc, MPI_COMM_WORLD);
        SqMatArray23Xcd selfenmom(2, 3, nc, MPI_COMM_WORLD);
        if (prank == 0) {
            std::ifstream fin("selfenergy_dyn.txt");
            if (fin.is_open()) {
                fin >> selfendyn;
                fin.close();
            }
            else std::cout << "Unable to open file" << std::endl;
            fin.clear();  // Clear flags
            fin.open("selfenergy_var.txt");
            if (fin.is_open()) {
                fin >> selfenvar;
                fin.close();
            }
            else std::cout << "Unable to open file" << std::endl;
            fin.clear();  // Clear flags
            fin.open("selfenergy_static.txt");
            if (fin.is_open()) {
                fin >> selfenstatic;
                fin.close();
            }
            else std::cout << "Unable to open file" << std::endl;
            fin.clear();  // Clear flags
            fin.open("selfenergy_moms.txt");
            if (fin.is_open()) {
                fin >> selfenmom;
                fin.close();
            }
            else std::cout << "Unable to open file" << std::endl;
        }
        selfendyn.broadcast(0);
        selfenvar.broadcast(0);
        selfenstatic.broadcast(0);
        selfenmom.broadcast(0);
        
        //pade.build(selfendyn, beta, Eigen::ArrayXi::LinSpaced(ndatalens, mindatalen, maxdatalen),
        //           Eigen::ArrayXi::LinSpaced(nstartfreqs, minstartfreq, maxstartfreq),
        //           Eigen::ArrayXi::LinSpaced(ncoefflens, mincoefflen, maxcoefflen), MPI_COMM_WORLD);
        Eigen::ArrayXd mats_freq = Eigen::ArrayXd::LinSpaced(selfendyn.dim1(), M_PI / beta, (2 * selfendyn.dim1() - 1) * M_PI / beta);
        mqem.assembleKernelMatrix(mats_freq, n_lrealfreq, lrealfreq, n_mrealfreq, rrealfreq, n_rrealfreq);
        mqem.computeSpectra(mats_freq, selfendyn, selfenvar, selfenmom);
        
        //pade.computeSpectra(selfenstatic, *H0, nenergies, minenergy, maxenergy, delenergy, physonly);
        mqem.computeRetardedFunc(selfenstatic);
        
        //en_idel = mqem.realFreqGrid() + Eigen::ArrayXcd::Constant(mqem.realFreqGrid().size(), 1i * delenergy);
        sigmaxx = longitConduc(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector());
        if (computesigmaxy) sigmaxy = hallConduc(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector());
        
        computeLattGFfCoeffs(*H0, mqem.retardedFunc(), mqem.realFreqGrid(), spectra);
        for (std::size_t i = 0; i < spectramastpart.size(); ++i) spectramastpart[i] = (spectramastpart[i] - spectramastpart[i].adjoint().eval()) / (2i * M_PI);
        spectramastpart.allGather();
        
        if (prank == 0) {
            //std::cout << "#spectra: " << pade.nPhysSpectra().sum() << std::endl;
            std::cout << "Optimal alpha for spin up: " << std::pow(10.0, mqem.optimalLog10alpha()(0)) << std::endl;
            std::cout << "sigmaxx = " << sigmaxx << std::endl;
            if (computesigmaxy) std::cout << "sigmaxy = " << sigmaxy << std::endl;
            printData("real_freqs.txt", mqem.realFreqGrid());
            printData("selfenergy_retarded.txt", mqem.retardedFunc());
            printData("spectramatrix.txt", spectra);
            printData("log10chi2_log10alpha.txt", mqem.log10chi2Log10alpha(0), std::numeric_limits<double>::max_digits10);
        }
        
        MPI_Finalize();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<60> > duration = stop - start;   // Unit is hours
        if (prank == 0) std::cout << "Execution time: " << duration.count() << " minutes" << std::endl;
        return 0;
    }
    
    
    if (measurewhat == "S") ntau4eiwt = 0;  // Not allocate eiwt array anyway if measuring S
    auto G0 = std::make_shared<BareGreenFunction>(beta, nc, nfcut, ntau, ntau4eiwt, MPI_COMM_WORLD);
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

    auto G = std::make_shared<GreenFunction>(beta, nc, nfcut, ntau, nbins4S, MPI_COMM_WORLD);
    
    //MPI_Barrier(MPI_COMM_WORLD);
    //usleep(1000 * prank);
    //std::this_thread::sleep_for(std::chrono::milliseconds(prank));
    //std::cout << "rank " << prank << " of " << psize << ": mastered size = " << G->fourierCoeffs().mastFlatPart().size() << ", mastered start = "
    //<< G->fourierCoeffs().mastFlatPart().start() << std::endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    //sleep(1);   // Wait 1 s
    //std::this_thread::sleep_for(std::chrono::seconds(1));
    //if (prank == 0) std::cout << sep << std::endl;

    std::shared_ptr<ImpurityProblem> impproblem;
    if (loc_corr) impproblem = std::make_shared<ImpurityProblem>(H0dec, G0, U, K, G);
    else impproblem = std::make_shared<ImpurityProblem>(H0, G0, U, K, G);

    CTAUXImpuritySolver impsolver(impproblem);
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
    dmft.parameters.at("convergence type") = converge_type;
    dmft.parameters.at("convergence criterion") = converge_criterion;
    dmft.parameters.at("local correlation") = loc_corr;
    
    
    
    std::pair<bool, double> converg;
    double interr;
    // double varmax, var;
    auto tstart = std::chrono::high_resolution_clock::now();
    auto tend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<60> > tdur;  // Unit is minutes
//    std::array<std::size_t, 2> so;
//    std::complex<double> zeta, zetasq;
//    double sgn;
    
    // Initialize self-energy and Green's functions
    if (ansatz == "insulator") {
        auto G0wmastpart = G0->fourierCoeffs().mastFlatPart();
        std::array<std::size_t, 2> so;
        if (loc_corr)
            for (std::size_t i = 0; i < G0wmastpart.size(); ++i) {
                so = G0wmastpart.global2dIndex(i);
                // Hybridization is set to zero to indicate the insulating ansatz (insulating bath should not screen the impurity)
                G0wmastpart[i].noalias() = -((1i * G0->matsubFreqs()(so[1]) + mu_eff) * Eigen::MatrixXcd::Identity(nc, nc) - H0dec->moments()(so[0], 0)).inverse();
            }
        else
            for (std::size_t i = 0; i < G0wmastpart.size(); ++i) {
                so = G0wmastpart.global2dIndex(i);
                // Hybridization is set to zero to indicate the insulating ansatz (insulating bath should not screen the impurity)
                G0wmastpart[i].noalias() = -((1i * G0->matsubFreqs()(so[1]) + mu_eff) * Eigen::MatrixXcd::Identity(nc, nc) - H0->moments()(so[0], 0)).inverse();
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
//        for (std::size_t i = 0; i < G->fourierCoeffs().mastPartSize(); ++i) {
//            so = G->fourierCoeffs().index2DinPart(i);  // Get the index in (spin, omega) space w.r.t. the full-sized data
//            zeta = 1i * ((2 * so[1] + 1) * M_PI / beta) + H0->mu;
//            zetasq = zeta * zeta + 1e-6i;
//            sgn = zetasq.imag() > 0 ? 1.0 : -1.0;
//            G->fourierCoeffs().masteredPart(i)(0, 0) = -0.5 * (zeta - sgn * std::sqrt(zetasq - 4.0));
//        }
//    }
    
    mqem.assembleKernelMatrix(G->matsubFreqs(), n_lrealfreq, lrealfreq, n_mrealfreq, rrealfreq, n_rrealfreq);
    if (prank == 0) printData("real_freqs.txt", mqem.realFreqGrid());
    
    bool computesigma;
    
    const int cw = 13;
    std::string dash(cw, '-');
    std::ofstream fiter;
    if (prank == 0) {
        fiter.open("iterations.txt", std::fstream::out | std::fstream::trunc);
        // std::cout << "\u03A3 \u03C9" << std::endl;
        if (measurewhat == "S") {
            fiter << std::setw(cw / 2 + 1) << " iter" << std::setw(cw + 1) << " converg" << std::setw(cw + 1) << " <order>" << std::setw(cw + 1)
            << " Im<S0>/w0" << std::setw(2 * cw + 2) << " var<n> / <n>" << std::setw(cw + 1) << " <sign>" << std::setw(cw + 1) << " <n> int err"
            << std::setw(cw + 1) << " sxx (e^2/h)";
            if (computesigmaxy) fiter << std::setw(cw + 1) << " sxy (e^2/h)";
            fiter << std::endl;
            fiter << " "  << std::string(cw / 2, '-') << " "               << dash       << " "               << dash       << " "
            << dash         << " "      << std::string(2 * cw + 1, '-') << " "               << dash      << " "               << dash
            << " "               << dash;
            if (computesigmaxy) fiter << " "               << dash;
            fiter << std::endl;
        }
        else if (measurewhat == "G") {
            fiter << std::setw(cw / 2 + 1) << " iter" << std::setw(cw + 1) << " converg" << std::setw(cw + 1) << " <order>" << std::setw(cw + 1)
            << " Im<S0>/w0" << std::setw(2 * cw + 2) << " var<n> / <n>" << std::setw(cw + 1) << " <sign>" << std::setw(cw + 1) << " sxx (e^2/h)";
            if (computesigmaxy) fiter << std::setw(cw + 1) << " sxy (e^2/h)";
            fiter << std::endl;
            fiter << " "  << std::string(cw / 2, '-') << " "               << dash       << " "               << dash       << " "
            << dash         << " "      << std::string(2 * cw + 1, '-') << " "               << dash      << " "               << dash;
            if (computesigmaxy) fiter << " "               << dash;
            fiter << std::endl;
        }
        fiter << std::scientific << std::setprecision(5);
    }
    do {
        dmft.incrementIter();
        
        if (prank == 0) std::cout << "DMFT iteration " << dmft.numIterations() << ":" << std::endl;
        
        if (prank == 0) std::cout << "    Impurity solver starts solving..." << std::endl;
        tstart = std::chrono::high_resolution_clock::now();
        interr = impsolver.solve();
        tend = std::chrono::high_resolution_clock::now();
        tdur = tend - tstart;
        if (prank == 0) std::cout << "    Impurity solver completed solving in " << tdur.count() << " minutes" << std::endl;
        
        dmft.approxSelfEnergy();
        
        dmft.updateLatticeGF();
        
        dmft.updateBathGF();
        
        converg = dmft.checkConvergence();
        computesigma = !computecondonce || (computecondonce && converg.first);
        
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
            computeLattGFfCoeffs(*H0, mqem.retardedFunc(), mqem.realFreqGrid(), spectra);
            for (std::size_t i = 0; i < spectramastpart.size(); ++i) spectramastpart[i] = (spectramastpart[i] - spectramastpart[i].adjoint().eval()) / (2i * M_PI);
            spectramastpart.allGather();
            tend = std::chrono::high_resolution_clock::now();
            tdur = tend - tstart;
            if (prank == 0) {
                std::cout << "    MQEM completed analytic continuation in " << tdur.count() << " minutes" << std::endl;
                std::cout << "    Optimal alpha for spin up: " << std::pow(10.0, mqem.optimalLog10alpha()(0)) << std::endl;
                //std::cout << "    #spectra = " << pade.nPhysSpectra()(0) << " (up), " << pade.nPhysSpectra()(1) << " (down)" << std::endl;
            }
            
            if (prank == 0) std::cout << "    Start computing conductivities..." << std::endl;
            tstart = std::chrono::high_resolution_clock::now();
            sigmaxx = longitConduc(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector());
            if (computesigmaxy) sigmaxy = hallConduc(*H0, mqem.retardedFunc(), beta, mqem.realFreqGrid(), mqem.realFreqIntVector());
            tend = std::chrono::high_resolution_clock::now();
            tdur = tend - tstart;
            if (prank == 0) std::cout << "    Computed conductivities in " << tdur.count() << " minutes" << std::endl;
        }  // Main iteration stops here
        
        // Output results
        //G->fourierCoeffs().mastFlatPart().allGather();
        if (prank == 0) {
            printData("G0.txt", G0->valsOnTauGrid());
            printData("G0matsubara.txt", G0->fourierCoeffs());
            printData("G.txt", G->valsOnTauGrid());
            printData("Gmatsubara.txt", G->fourierCoeffs());
            printData("selfenergy_dyn.txt", dmft.dynSelfEnergy(), std::numeric_limits<double>::max_digits10);
            printData("selfenergy_var.txt", dmft.selfEnergyVar(), std::numeric_limits<double>::max_digits10);
            printData("selfenergy_static.txt", dmft.staticSelfEnergy(), std::numeric_limits<double>::max_digits10);
            printData("selfenergy_moms.txt", dmft.selfEnergyMoms(), std::numeric_limits<double>::max_digits10);
            if (computesigma) {
                printData("selfenergy_retarded.txt", mqem.retardedFunc());
                printData("spectramatrix.txt", spectra);
                printData("log10chi2_log10alpha.txt", mqem.log10chi2Log10alpha(0), std::numeric_limits<double>::max_digits10);
            }
            printHistogram("histogram.txt", impsolver.vertexOrderHistogram());
        }
        
        if (prank == 0) {
            if (measurewhat == "S") {
                fiter << " " << std::setw(cw / 2) << dmft.numIterations() << " " << std::setw(cw) << converg.second << " " << std::setw(cw)
                << impsolver.aveVertexOrder() << " " << std::setw(cw) << std::imag(dmft.dynSelfEnergy()(0, 0, 0, 0) + dmft.staticSelfEnergy()(0, 0, 0, 0)) / (M_PI / beta)
                << " " << std::setw(cw) << G->elecDensStdDev()(0, 0) << " " << std::setw(cw) << G->elecDensities().sum() << " " << std::setw(cw) << impsolver.fermiSign()
                << " " << std::setw(cw) << interr;
            }
            else if (measurewhat == "G") {
                fiter << " " << std::setw(cw / 2) << dmft.numIterations() << " " << std::setw(cw) << converg.second << " " << std::setw(cw)
                << impsolver.aveVertexOrder() << " " << std::setw(cw) << std::imag(dmft.dynSelfEnergy()(0, 0, 0, 0) + dmft.staticSelfEnergy()(0, 0, 0, 0)) / (M_PI / beta)
                << " " << std::setw(cw) << G->elecDensStdDev()(0, 0) << " " << std::setw(cw) << G->elecDensities().sum() << " " << std::setw(cw) << impsolver.fermiSign();
            }
            if (computesigma) {
                fiter << " " << std::setw(cw) << sigmaxx;
                if (computesigmaxy) fiter << " " << std::setw(cw) << sigmaxy;
            }
            else {
                fiter << " " << std::setw(cw) << "--";
                if (computesigmaxy) fiter << " " << std::setw(cw) << "--";
            }
            fiter << std::endl;
        }
    }
    while (!converg.first && dmft.numIterations() < nitmax);
    if (!converg.first && prank == 0) {
        fiter << sep << std::endl;
        fiter << ">>>>>> DMFT self-consistency iteration did not converge! <<<<<<" << std::endl;
    }
    if (prank == 0) {
        fiter.close();
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
