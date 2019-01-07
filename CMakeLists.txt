cmake_minimum_required(VERSION 2.8.3)
project(gn_solver)

find_package(catkin REQUIRED COMPONENTS exotica_core)

find_package(Eigen3 REQUIRED)

AddInitializer(GNsolver)
GenInitializers()

catkin_package(
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS exotica_core
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} src/GaussNewton.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_initializers ${catkin_EXPORTED_TARGETS})

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(FILES exotica_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
