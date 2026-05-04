#pragma once
#include <iostream>
#include <cmath>

inline int& g_passed() { static int n = 0; return n; }
inline int& g_failed() { static int n = 0; return n; }

#define ASSERT_TRUE(expr) \
    do { if (!(expr)) { ++g_failed(); \
        std::cout << "  [FAIL] " #expr \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; } \
    else { ++g_passed(); } } while(0)

#define ASSERT_NEAR(a,b,eps) \
    do { if (std::fabs((double)(a)-(double)(b))>(eps)) { ++g_failed(); \
        std::cout << "  [FAIL] |" << (double)(a) << " - " << (double)(b) \
                  << "| > " << (eps) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; } \
    else { ++g_passed(); } } while(0)
