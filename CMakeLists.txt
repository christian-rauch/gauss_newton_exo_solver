cmake_minimum_required(VERSION 2.8.3)
project(gn_solver)

find_package(catkin REQUIRED COMPONENTS exotica)

find_package(Eigen3 REQUIRED)

AddInitializer(GNsolver)
GenInitializers()

catkin_package(
    LIBRARIES gn_solver
    CATKIN_DEPENDS exotica
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(gn_solver src/GaussNewton.cpp)
target_link_libraries(gn_solver ${catkin_LIBRARIES})
add_dependencies(gn_solver gn_solver_initializers)

install(TARGETS gn_solver
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(FILES exotica_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
