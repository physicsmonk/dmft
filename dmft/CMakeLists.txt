set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
#find_package(GMP REQUIRED)
#find_package(MPFR REQUIRED)

# Add dependent library
add_subdirectory(dependencies/pugixml)

add_library(dmft
    src/bare_hamiltonian.cpp
    src/green_function.cpp
    src/ct_aux_imp_solver.cpp
    src/self_consistency.cpp
    )

target_include_directories(dmft PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies
    )

target_link_libraries(dmft PUBLIC
    Eigen3::Eigen
    #${MPFR_LIBRARIES}
    #${GMP_LIBRARIES}
    MPI::MPI_CXX
    pugixml
    )
