from helper_functions import initialize_const
#Test variables
test_k = 5
test_Nr = [8]*test_k
test_Nt = [8]*test_k

test_n = [600, 100, 200, 400, 500]
test_l = [10, 20, 40, 50, 100]
test_snr_db = [10]*test_k
test_Pt = [32] * test_k
test_fs = [2] * test_k
test_B = [1000] *test_k
test_epsilon = [1e-2, 1e-3 ,1e-3, 1e-4, 1e-5]
TEST_CONST = initialize_const(B = test_B, Pt = test_Pt, fs = test_fs, snr_db = test_snr_db, desired_CNR = [], Nt = test_Nt, Nr = test_Nr, K = test_k, N = test_n, L = test_l, epsilon = test_epsilon)

if __name__ == "__main__":
    print(TEST_CONST)