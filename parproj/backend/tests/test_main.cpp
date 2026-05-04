#include "test_harness.h"

void register_csr_tests();
void register_feature_tests();
void register_ml_tests();

int main() {
    std::cout << "\n====== Graph ML Pipeline — Unit Tests ======\n\n";
    register_csr_tests();
    register_feature_tests();
    register_ml_tests();
    std::cout << "\n============================================\n";
    std::cout << "  Passed: " << g_passed()
              << "  Failed: " << g_failed() << "\n";
    std::cout << "============================================\n";
    return g_failed() > 0 ? 1 : 0;
}
