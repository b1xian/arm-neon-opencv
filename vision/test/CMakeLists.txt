set(UT_NAME va_cv_ut)


file(GLOB_RECURSE test_srcs LIST_DIRECTORIES FALSE
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

file(GLOB_RECURSE test_heads LIST_DIRECTORIES FALSE
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
                        
add_executable(${UT_NAME} ${test_srcs})
target_compile_options(${UT_NAME} PRIVATE -std=c++14)
target_compile_options(${UT_NAME} PRIVATE -O3 )

target_include_directories(${UT_NAME}
                            PRIVATE
                            ${test_heads}
                            ${OPENCV_INCLUDE_DIR}
                            )

target_link_libraries(${UT_NAME}
                        PRIVATE
                        ${OPENCV_LIB}
                        vacv
                        )
message(${CMAKE_BINARY_DIR})
message(${CMAKE_CURRENT_SOURCE_DIR})
message(${TARGET_OS})
message(${CMAKE_BINARY_DIR}/vacv_test/)
install(FILES ${CMAKE_BINARY_DIR}/vision/test/${UT_NAME} DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/176x144.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/284x214.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/640x360.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/1280x720.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/1280x720_grey.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/1920x1080.jpeg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/2560x1440.jpeg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
if (TARGET_OS MATCHES osx)
    install(FILES ${CMAKE_BINARY_DIR}/libvacv.dylib DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/)
else()
    install(FILES ${CMAKE_BINARY_DIR}/libvacv.so DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/)
endif()
