cmake_minimum_required(VERSION 3.15)

project(x-voice-ggml-cpp
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build X-Voice.cpp dependencies as static libraries" FORCE)

set(XVOICE_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml" CACHE PATH "Path to vendored ggml")
set(XVOICE_CPP_PINYIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-pinyin" CACHE PATH "Path to vendored cpp-pinyin")
set(XVOICE_CPP_PINYIN_DICT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/resources/zh_pinyin_dict" CACHE PATH "Path to the cpp-pinyin dictionary root used by the zh frontend")
set(XVOICE_CPPJIEBA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cppjieba" CACHE PATH "Path to vendored cppjieba")
set(XVOICE_NLOHMANN_JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann-json" CACHE PATH "Path to vendored nlohmann/json")
option(XVOICE_ALLOW_WORKBENCH_GGML_FALLBACK "Allow local ggbond workbench ggml fallback when vendor/ggml is missing" OFF)
set(XVOICE_ENABLE_ZH_FRONTEND "AUTO" CACHE STRING "Enable Chinese plain-text frontend with cpp-pinyin/cppjieba: AUTO, ON, or OFF")
set_property(CACHE XVOICE_ENABLE_ZH_FRONTEND PROPERTY STRINGS AUTO ON OFF)

if(NOT EXISTS "${XVOICE_GGML_DIR}/CMakeLists.txt")
    set(XVOICE_WORKBENCH_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/ggml")
    if(XVOICE_ALLOW_WORKBENCH_GGML_FALLBACK AND EXISTS "${XVOICE_WORKBENCH_GGML_DIR}/CMakeLists.txt")
        message(STATUS "vendor/ggml is missing; using workbench ggml at ${XVOICE_WORKBENCH_GGML_DIR}")
        set(XVOICE_GGML_DIR "${XVOICE_WORKBENCH_GGML_DIR}")
    else()
        message(FATAL_ERROR "vendor/ggml is missing; run git submodule update --init --recursive, pass -DXVOICE_GGML_DIR=/path/to/ggml, or set -DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON for local ggbond development.")
    endif()
endif()

add_subdirectory("${XVOICE_GGML_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

add_library(x_voice_ggml_cpp STATIC
    src/audio.cpp
    src/frontend.cpp
    src/model.cpp
    src/srp.cpp
    src/stage2.cpp
    src/synthesis.cpp
    src/vocos.cpp
    src/x_voice.cpp
)

target_include_directories(x_voice_ggml_cpp
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${XVOICE_GGML_DIR}/include"
        "${XVOICE_GGML_DIR}/src"
)

target_link_libraries(x_voice_ggml_cpp PUBLIC ggml)

set(XVOICE_ZH_FRONTEND_DEPS_AVAILABLE OFF)
if(EXISTS "${XVOICE_CPP_PINYIN_DIR}/CMakeLists.txt" AND EXISTS "${XVOICE_CPPJIEBA_DIR}/CMakeLists.txt")
    set(XVOICE_ZH_FRONTEND_DEPS_AVAILABLE ON)
endif()

set(XVOICE_ZH_FRONTEND_ENABLED OFF)
if(XVOICE_ENABLE_ZH_FRONTEND STREQUAL "ON")
    if(NOT XVOICE_ZH_FRONTEND_DEPS_AVAILABLE)
        message(FATAL_ERROR "XVOICE_ENABLE_ZH_FRONTEND=ON requires cpp-pinyin at ${XVOICE_CPP_PINYIN_DIR} and cppjieba at ${XVOICE_CPPJIEBA_DIR}")
    endif()
    set(XVOICE_ZH_FRONTEND_ENABLED ON)
elseif(XVOICE_ENABLE_ZH_FRONTEND STREQUAL "AUTO")
    if(XVOICE_ZH_FRONTEND_DEPS_AVAILABLE)
        set(XVOICE_ZH_FRONTEND_ENABLED ON)
    endif()
elseif(NOT XVOICE_ENABLE_ZH_FRONTEND STREQUAL "OFF")
    message(FATAL_ERROR "XVOICE_ENABLE_ZH_FRONTEND must be AUTO, ON, or OFF")
endif()

if(XVOICE_ZH_FRONTEND_ENABLED)
    if(NOT EXISTS "${XVOICE_CPP_PINYIN_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "XVOICE_ENABLE_ZH_FRONTEND requires cpp-pinyin at ${XVOICE_CPP_PINYIN_DIR}")
    endif()
    if(NOT EXISTS "${XVOICE_CPPJIEBA_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "XVOICE_ENABLE_ZH_FRONTEND requires cppjieba at ${XVOICE_CPPJIEBA_DIR}")
    endif()

    set(CPP_PINYIN_BUILD_TESTS OFF CACHE BOOL "Build cpp-pinyin tests" FORCE)
    set(CPP_PINYIN_INSTALL OFF CACHE BOOL "Install cpp-pinyin" FORCE)
    add_subdirectory("${XVOICE_CPP_PINYIN_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cpp-pinyin")
    target_link_libraries(x_voice_ggml_cpp PUBLIC cpp-pinyin::cpp-pinyin)

    set(CPPJIEBA_TOP_LEVEL_PROJECT OFF CACHE BOOL "Build cppjieba tests" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_limonp TRUE)
    add_subdirectory("${XVOICE_CPPJIEBA_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cppjieba")
    target_link_libraries(x_voice_ggml_cpp PUBLIC cppjieba)

    target_compile_definitions(x_voice_ggml_cpp
        PRIVATE
            XVOICE_ENABLE_ZH_FRONTEND=1
            XVOICE_DEFAULT_CPP_PINYIN_DICT="${XVOICE_CPP_PINYIN_DICT_ROOT}"
            XVOICE_DEFAULT_CPPJIEBA_DICT="${XVOICE_CPPJIEBA_DIR}/dict"
            XVOICE_DEFAULT_ZH_OVERRIDE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/zh_frontend_overrides.tsv"
    )
    message(STATUS "X-Voice zh frontend: enabled (cpp-pinyin=${XVOICE_CPP_PINYIN_DIR}, dict=${XVOICE_CPP_PINYIN_DICT_ROOT}, cppjieba=${XVOICE_CPPJIEBA_DIR})")
else()
    message(STATUS "X-Voice zh frontend: disabled")
endif()

if(EXISTS "${XVOICE_NLOHMANN_JSON_DIR}/single_include/nlohmann/json.hpp")
    add_executable(xvoice-merge-pinyin-dicts tools/merge_pinyin_dicts.cpp)
    target_include_directories(xvoice-merge-pinyin-dicts PRIVATE "${XVOICE_NLOHMANN_JSON_DIR}/single_include")
endif()

add_executable(xvoice-quantize-gguf tools/quantize_gguf.cpp)
target_include_directories(xvoice-quantize-gguf PRIVATE
    "${XVOICE_GGML_DIR}/include"
    "${XVOICE_GGML_DIR}/src"
)
target_link_libraries(xvoice-quantize-gguf PRIVATE ggml)

add_executable(x-voice-cli src/x_voice_cli.cpp)
target_link_libraries(x-voice-cli PRIVATE x_voice_ggml_cpp)

install(TARGETS x_voice_ggml_cpp x-voice-cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY resources/
    DESTINATION share/x-voice-ggml-cpp/resources
)

install(DIRECTORY docs/
    DESTINATION share/x-voice-ggml-cpp/docs
)

install(DIRECTORY scripts/
    DESTINATION share/x-voice-ggml-cpp/scripts
    USE_SOURCE_PERMISSIONS
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)
