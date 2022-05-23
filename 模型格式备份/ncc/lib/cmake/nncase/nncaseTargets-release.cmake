#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nncase" for configuration "Release"
set_property(TARGET nncase APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nncase PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/nncase.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/nncase.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS nncase )
list(APPEND _IMPORT_CHECK_FILES_FOR_nncase "${_IMPORT_PREFIX}/lib/nncase.lib" "${_IMPORT_PREFIX}/bin/nncase.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
