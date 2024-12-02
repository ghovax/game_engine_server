import os
import ctypes.util

# Try multiple library paths
LIBRARY_PATHS = [
    "/opt/homebrew/Cellar/assimp/5.4.3/lib/libassimp.5.4.3.dylib",
    "/opt/homebrew/lib/libassimp.dylib",
    "/usr/local/lib/libassimp.dylib",
    "/usr/lib/libassimp.dylib",
]


def configure_assimp():
    # Set multiple environment variables to help find the library
    for lib_path in LIBRARY_PATHS:
        if os.path.exists(lib_path):
            os.environ["ASSIMP_LIBRARY_PATH"] = lib_path
            os.environ["DYLD_LIBRARY_PATH"] = os.path.dirname(lib_path)
            break

    # Also try to use ctypes to find the library
    assimp_path = ctypes.util.find_library("assimp")
    if assimp_path:
        os.environ["ASSIMP_PATH"] = assimp_path
