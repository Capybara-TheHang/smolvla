cmake_minimum_required(VERSION 3.5)

project(NTH_POC2_2_URDF_20251119v1)

find_package(ament_cmake REQUIRED)

foreach(dir config launch)
  install(DIRECTORY ${dir}/
    DESTINATION share/${PROJECT_NAME}/${dir})
endforeach()

install(FILES
  "north_poc2_2_v3_1.urdf"
  "north_poc2_2_v3_1_fixbody.urdf"
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
