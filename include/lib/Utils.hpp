#pragma once

#ifdef DEBUG
    #define LOG(n)   std::cout << "\033[1;32m[LOG] " << n << "\033[0m" << std::endl;
    #define FIX(n)   std::cout << "\033[1;33m[FIX]   " << n << "\033[0m" << std::endl;
    #define INFO(n)  std::cout << "[INFO] " << n << std::endl;
    #define ERROR(n) std::cerr << "\033[1;31m[ERROR] " << n << "\033[0m" << std::endl;
#else
    #define LOG(n)
    #define FIX(n)
    #define INFO(n)
    #define ERROR(n)
#endif
