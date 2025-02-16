import math
import random
import time

import torch
from sympy import isprime
from memory_profiler import profile

class NTT:

    def __init__(self, use_cuda=True):

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def isInteger(self, M):

        return type(M).__name__ == 'int'

    def isPrime(self, M):

        assert self.isInteger(M), 'Not an integer.'
        return isprime(M)


    def modExponent(self, base, power, M):

        base_t = torch.tensor(base, dtype=torch.int64, device=self.device)
        power_t = torch.tensor(power, dtype=torch.int64, device=self.device)
        M_t = torch.tensor(M, dtype=torch.int64, device=self.device)

        result_t = torch.ones((), dtype=torch.int64, device=self.device)
        while power_t.item() > 0:

            if (power_t & 1).item() == 1:
                result_t = (result_t * base_t) % M_t
            base_t = (base_t * base_t) % M_t
            power_t = power_t >> 1

        return result_t.item()


    def modInv(self, x, M):

        x_t = torch.tensor(x, dtype=torch.int64, device=self.device)
        M_t = torch.tensor(M, dtype=torch.int64, device=self.device)

        t = torch.tensor(0, dtype=torch.int64, device=self.device)
        new_t = torch.tensor(1, dtype=torch.int64, device=self.device)
        r = M_t.clone()
        new_r = x_t.clone()

        while new_r.item() != 0:
            quotient = r // new_r
            t, new_t = new_t, t - quotient * new_t
            r, new_r = new_r, r % new_r

        if r.item() > 1:
            return "x is not invertible."

        if t.item() < 0:
            t = t + M_t

        return t.item()


    def existSmallN(self, r, M, N):

        r_t = torch.tensor(r, dtype=torch.int64, device=self.device)
        M_t = torch.tensor(M, dtype=torch.int64, device=self.device)

        if N < 3:
            return False

        a = torch.empty(N - 1, dtype=torch.int64, device=self.device)
        a[0] = r_t % M_t
        for i in range(1, N - 1):
            a[i] = (a[i - 1] * r_t) % M_t

        if (a[1:] == 1).any().item():
            return True
        return False


    def NthRootOfUnity(self, M, N):

        assert self.isPrime(M), 'Not a prime.'
        assert (M - 1) % N == 0, 'N cannot divide phi(M)'

        phi_M = M - 1
        while True:
            alpha = random.randrange(1, M)
            beta = self.modExponent(alpha, phi_M // N, M)
            if not self.existSmallN(beta, M, N):
                return beta


    def bitReverse(self, num, length_):

        rev_num = 0
        for i in range(length_):
            if (num >> i) & 1:
                rev_num |= 1 << (length_ - 1 - i)
        return rev_num


    def orderReverse(self, poly, N_bit):

        N = poly.shape[0]

        idx_list = []
        for i in range(N):
            idx_list.append(self.bitReverse(i, N_bit))
        idx_t = torch.tensor(idx_list, dtype=torch.long, device=self.device)

        return poly[idx_t]


    def ntt(self, poly, M, N, w):


        poly_t = torch.tensor(poly, dtype=torch.int64, device=self.device)
        M_t = torch.tensor(M, dtype=torch.int64, device=self.device)
        w_t = torch.tensor(w, dtype=torch.int64, device=self.device)

        N_bit = N.bit_length() - 1
        poly_t = self.orderReverse(poly_t, N_bit)

        length = 1
        for i in range(N_bit):

            m = 1 << (i + 1)
            half_m = m >> 1

            wj = torch.empty(half_m, dtype=torch.int64, device=self.device)
            wj[0] = torch.tensor(1, dtype=torch.int64, device=self.device)
            
            w_exp = self.modExponent(w, N // m, M)

            w_pow = torch.tensor(w_exp, dtype=torch.int64, device=self.device)
            for j in range(1, half_m):
                wj[j] = (wj[j-1] * w_pow) % M_t


            block_starts = torch.arange(0, N, m, device=self.device)
            for start in block_starts:

                idx_range = torch.arange(half_m, device=self.device)
                idx1 = start + idx_range
                idx2 = start + idx_range + half_m

                even = poly_t[idx1]
                odd  = (poly_t[idx2] * wj[idx_range]) % M_t


                poly_t[idx1] = (even + odd) % M_t
                poly_t[idx2] = (even - odd) % M_t

            length <<= 1

        return poly_t


    def intt(self, points, M, N, w):

        inv_w = self.modInv(w, M)
        if isinstance(inv_w, str):
            raise ValueError(inv_w)

        inv_N = self.modInv(N, M)
        if isinstance(inv_N, str):
            raise ValueError(inv_N)

        points_t = torch.tensor(points, dtype=torch.int64, device=self.device)
        points_t = self.ntt(points_t, M, N, inv_w)

        M_t = torch.tensor(M, dtype=torch.int64, device=self.device)
        inv_N_t = torch.tensor(inv_N, dtype=torch.int64, device=self.device)
        points_t = (points_t * inv_N_t) % M_t

        return points_t

@profile
def negacyclic_convolution_test():

    ntt = NTT(use_cuda=True)

    M = 8380417
    A_list = list(range(1, 2049))
    B_list = list(range(5, 2053))
    n = len(A_list)

    print(f"Modulus = {M}")
    print("Polynomial A =", A_list[:8], "... (total length =", len(A_list), ")")
    print("Polynomial B =", B_list[:8], "... (total length =", len(B_list), ")")
    print(f"Target ring: Z_{M}[x]/(x^{n}+1)  (negacyclic)")

    size = 2 * n
    A_ext = A_list + [0]*(size - n)
    B_ext = B_list + [0]*(size - n)

    w_2n = ntt.NthRootOfUnity(M, size)
    print(f"\n2n-th root of unity = {w_2n},  2n={size}")
    wn = ntt.modExponent(w_2n, n, M)
    print(f"w_2n^{n} mod M = {wn} (expect {M-1} for negacyclic, i.e. -1 mod M)")

    a = time.time()
    A_ntt = ntt.ntt(A_ext, M, size, w_2n)
    B_ntt = ntt.ntt(B_ext, M, size, w_2n)
    b = time.time()
    print("NTT time: ", round(b - a, 6))


    c = time.time()
    AB_ntt = (A_ntt * B_ntt) % M
    AB_time_2n = ntt.intt(AB_ntt, M, size, w_2n)
    C_negacyclic = (AB_time_2n[:n] - AB_time_2n[n:2*n]) % M
    d = time.time()
    print("INTT time: ", round(d - c, 6))

    print("\n--- Frequency domain data (first 8) ---")
    print("A_ntt[:8] =", A_ntt[:8].tolist())
    print("B_ntt[:8] =", B_ntt[:8].tolist())
    print("AB_ntt[:8] =", AB_ntt[:8].tolist())

    print("\n--- Time domain raw (2n) after iNTT (first 8) ---")
    print("AB_intt_2n[:8] =", AB_time_2n[:8].tolist())

    print("\n--- Final result mod (x^n+1) => negacyclic convolution (first 8) ---")
    print("C_negacyclic[:8] =", C_negacyclic[:8].tolist())

if __name__ == "__main__":
    negacyclic_convolution_test()
