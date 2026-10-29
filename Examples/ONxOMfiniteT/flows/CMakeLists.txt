set(ONxOMFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_ON/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_ON/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_ON/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_ON/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_ON/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(ONxOMFlows STATIC ${ONxOMFlows_SOURCES})
add_library(ONxOMFlows_nowarn INTERFACE)
target_compile_options(ONxOMFlows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(ONxOMFlows DiFfRG::DiFfRG ONxOMFlows_nowarn)
