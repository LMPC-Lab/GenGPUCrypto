import math
import random
import torch
from sympy import isprime
import time
from memory_profiler import profile

class NTT:

    def __init__(self, device=None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def isInteger(self, M):
        return type(M).__name__ == 'int'

    def isPrime(self, M):
        assert self.isInteger(M), 'Not an integer.'
        return isprime(M)


    def modExponent(self, base, power, M):
        device = self.device
        result = torch.tensor(1, dtype=torch.int64, device=device)
        M_tensor = torch.tensor(M, dtype=torch.int64, device=device)
        base = torch.tensor(base % M, dtype=torch.int64, device=device)
        power = int(power)
        while power > 0:
            if power & 1:
                result = (result * base) % M_tensor
            base = (base * base) % M_tensor
            power //= 2
        return int(result.item())


    def modInv(self, x, M):
        device = self.device
        t = torch.tensor(0, dtype=torch.int64, device=device)
        new_t = torch.tensor(1, dtype=torch.int64, device=device)
        r = torch.tensor(M, dtype=torch.int64, device=device)
        new_r = torch.tensor(x, dtype=torch.int64, device=device)
        while new_r.item() != 0:
            quotient = (r // new_r).item()  
            t, new_t = new_t, t - quotient * new_t
            r, new_r = new_r, r % new_r
        if r.item() > 1:
            return "x is not invertible."
        if t.item() < 0:
            t += M
        return int(t.item())


    def existSmallN(self, r, M, N):
        for k in range(2, N):
            if self.modExponent(r, k, M) == 1:
                return True
        return False


    def NthRootOfUnity(self, M, N):
        assert self.isPrime(M), 'Not a prime.'
        assert (M - 1) % N == 0, 'N cannot divide phi(M)'
        phi_M = M - 1
        while True:
            alpha = random.randrange(1, M)

            beta = self.modExponent(alpha, phi_M / N, M)
            if not self.existSmallN(beta, M, N):
                return int(beta)


    def bitReverse(self, num, length_):
        rev_num = 0
        for i in range(length_):
            if (num >> i) & 1:
                rev_num |= 1 << (length_ - 1 - i)
        return rev_num

    def orderReverse(self, poly, N_bit):
        N = poly.shape[0]
        indices = torch.empty(N, dtype=torch.int64, device=self.device)
        for i in range(N):
            indices[i] = self.bitReverse(i, N_bit)
        poly = poly[indices]
        return poly

    def ntt(self, poly, M, N, w):
        device = self.device
        M_tensor = torch.tensor(M, dtype=torch.int64, device=device)

        if not isinstance(poly, torch.Tensor):
            poly = torch.tensor(poly, dtype=torch.int64, device=device)
        N_bit = N.bit_length() - 1
        poly = self.orderReverse(poly, N_bit)

        global pre_time1, pre_time2
        pre_time1= time.time()

        precomp = torch.empty(N, dtype=torch.int64, device=device)
        precomp[0] = 1
        for k in range(1, N):
            precomp[k] = (precomp[k - 1] * w) % M

        pre_time2 = time.time()

        for i in range(N_bit):
            shift = N_bit - 1 - i
            half = N // 2
            j = torch.arange(half, dtype=torch.int64, device=device)
            block_size = 1 << shift
            P = (j // block_size) * block_size
            twiddle = precomp[P]
            even = poly[0::2]
            odd = poly[1::2]
            new_first = (even + (odd * twiddle) % M_tensor) % M_tensor
            new_second = (even - (odd * twiddle) % M_tensor) % M_tensor
            poly = torch.cat([new_first, new_second])
        return poly

    def intt(self, points, M, N, w):
        device = self.device
        M_tensor = torch.tensor(M, dtype=torch.int64, device=device)
        inv_w = self.modInv(w, M)
        inv_N = self.modInv(N, M)
        poly = self.ntt(points, M, N, inv_w)
        poly = (poly * inv_N) % M_tensor
        return poly


pre_time1 = 0
pre_time2 = 0

@profile
def negacyclic_convolution_test():
    ntt_instance = NTT()
    device = ntt_instance.device
    M = 12289
    A = list(range(1, 1025))
    B = list(range(5, 1029))
    n = len(A)

    print(f"Modulus = {M}")
    print("Polynomial A (first 10) =", A[:10], "...")
    print("Polynomial B (first 10) =", B[:10], "...")
    print(f"Target ring: Z_{M}[x]/(x^{n}+1)  (negacyclic)")

    size = 2 * n
    A_ext = A + [0] * (size - n)
    B_ext = B + [0] * (size - n)

    w_2n = ntt_instance.NthRootOfUnity(M, size)
    print(f"\n2n-th root of unity = {w_2n},  2n = {size}")

    wn = ntt_instance.modExponent(w_2n, n, M)
    print(f"w_2n^{n} mod M = {wn} (expect -1 for negacyclic)")

    A_ext_tensor = torch.tensor(A_ext, dtype=torch.int64, device=device)
    B_ext_tensor = torch.tensor(B_ext, dtype=torch.int64, device=device)

    a = time.time()
    A_ntt = ntt_instance.ntt(A_ext_tensor.clone(), M, size, w_2n)
    t1 = pre_time2 - pre_time1
    B_ntt = ntt_instance.ntt(B_ext_tensor.clone(), M, size, w_2n)
    t2 = pre_time2 - pre_time1
    b = time.time()
    print("NTT time", round(b - a - t1 - t2 , 6))
    # print(t1, t2)
    c = time.time()
    M_tensor = torch.tensor(M, dtype=torch.int64, device=device)
    AB_ntt_tensor = (A_ntt * B_ntt) % M_tensor
    AB_ntt = AB_ntt_tensor.cpu().tolist()
    AB_time_2n_tensor = ntt_instance.intt(AB_ntt_tensor, M, size, w_2n)
    AB_time_2n = AB_time_2n_tensor.cpu().tolist()
    C_negacyclic = [(AB_time_2n[i] - AB_time_2n[i + n]) % M for i in range(n)]
    d = time.time()
    print("INTT time", round(d - c, 6))

    print("\n--- Frequency domain data (first 10) ---")
    print("A_ntt =", A_ntt.cpu().tolist())
    print("B_ntt =", B_ntt.cpu().tolist())
    print("AB_ntt =", AB_ntt)

    print("\n--- Time domain raw (2n) after iNTT (first 10) ---")
    print("AB_intt_2n =", AB_time_2n)

    print("\n--- Final negacyclic convolution result (first 11) ---")
    print("C_negacyclic =", C_negacyclic)


if __name__ == "__main__":
    negacyclic_convolution_test()
