# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/user1/.local/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/user1/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/user1/work/rocJPEG/samples/jpegDecode

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/user1/work/rocJPEG/samples/jpegDecode/build

# Include any dependencies generated for this target.
include CMakeFiles/jpegdecode.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/jpegdecode.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/jpegdecode.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/jpegdecode.dir/flags.make

CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o: CMakeFiles/jpegdecode.dir/flags.make
CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o: /home/user1/work/rocJPEG/samples/jpegDecode/jpegdecode.cpp
CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o: CMakeFiles/jpegdecode.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/user1/work/rocJPEG/samples/jpegDecode/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o"
	/opt/rocm/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o -MF CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o.d -o CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o -c /home/user1/work/rocJPEG/samples/jpegDecode/jpegdecode.cpp

CMakeFiles/jpegdecode.dir/jpegdecode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jpegdecode.dir/jpegdecode.cpp.i"
	/opt/rocm/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user1/work/rocJPEG/samples/jpegDecode/jpegdecode.cpp > CMakeFiles/jpegdecode.dir/jpegdecode.cpp.i

CMakeFiles/jpegdecode.dir/jpegdecode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jpegdecode.dir/jpegdecode.cpp.s"
	/opt/rocm/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user1/work/rocJPEG/samples/jpegDecode/jpegdecode.cpp -o CMakeFiles/jpegdecode.dir/jpegdecode.cpp.s

# Object files for target jpegdecode
jpegdecode_OBJECTS = \
"CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o"

# External object files for target jpegdecode
jpegdecode_EXTERNAL_OBJECTS =

jpegdecode: CMakeFiles/jpegdecode.dir/jpegdecode.cpp.o
jpegdecode: CMakeFiles/jpegdecode.dir/build.make
jpegdecode: /usr/lib/x86_64-linux-gnu/libva.so
jpegdecode: /usr/lib/x86_64-linux-gnu/libva-drm.so
jpegdecode: /usr/lib/x86_64-linux-gnu/libdrm.so
jpegdecode: /opt/rocm/hip/lib/libamdhip64.so.5.6.50600
jpegdecode: /opt/rocm-5.6.0/llvm/lib/clang/16.0.0/lib/linux/libclang_rt.builtins-x86_64.a
jpegdecode: CMakeFiles/jpegdecode.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/user1/work/rocJPEG/samples/jpegDecode/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable jpegdecode"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jpegdecode.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/jpegdecode.dir/build: jpegdecode
.PHONY : CMakeFiles/jpegdecode.dir/build

CMakeFiles/jpegdecode.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/jpegdecode.dir/cmake_clean.cmake
.PHONY : CMakeFiles/jpegdecode.dir/clean

CMakeFiles/jpegdecode.dir/depend:
	cd /home/user1/work/rocJPEG/samples/jpegDecode/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user1/work/rocJPEG/samples/jpegDecode /home/user1/work/rocJPEG/samples/jpegDecode /home/user1/work/rocJPEG/samples/jpegDecode/build /home/user1/work/rocJPEG/samples/jpegDecode/build /home/user1/work/rocJPEG/samples/jpegDecode/build/CMakeFiles/jpegdecode.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/jpegdecode.dir/depend

