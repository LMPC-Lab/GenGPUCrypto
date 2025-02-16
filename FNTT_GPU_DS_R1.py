import math
import torch
from sympy import isprime
import time
from memory_profiler import profile

class NTT_GPU:

    def isPrime(self, M):
        return isprime(M)

    def modExponent(self, base, power, M):
        if not isinstance(base, torch.Tensor):
            base = torch.tensor(base, dtype=torch.int64)
        if not isinstance(power, torch.Tensor):
            power = torch.tensor(power, dtype=torch.int64)

        device = base.device
        result = torch.ones_like(power, dtype=torch.int64).to(device)
        base = base % M
        power = power.clone().to(device)

        while torch.any(power > 0):
            mask = (power % 2 == 1)
            result[mask] = (result[mask] * base) % M
            base = (base * base) % M
            power = power // 2
        return result

    def modInv(self, x, M):
        return self.modExponent(x, M - 2, M)

    def existSmallN(self, r, M, N):
        if N <= 2:
            return torch.tensor(False, device=r.device)
        k = torch.arange(2, N, device=r.device)
        powers = self.modExponent(r, k, M)
        return torch.any(powers == 1)

    def NthRootOfUnity(self, M, N):
        phi_M = M - 1
        while True:
            alpha = torch.randint(1, M, (1,), dtype=torch.int64).item()
            beta = self.modExponent(alpha, phi_M // N, M)
            if not self.existSmallN(beta, M, N):
                return beta

    def bitReverseVector(self, nums, length_):
        rev_nums = torch.zeros_like(nums)
        for i in range(length_):
            rev_nums |= ((nums >> i) & 1) << (length_ - 1 - i)
        return rev_nums

    def orderReverse(self, poly, N_bit):
        n = poly.size(0)
        indices = torch.arange(n, device=poly.device)
        rev_indices = self.bitReverseVector(indices, N_bit)
        return poly[rev_indices].clone()

    def ntt(self, poly, M, N, w):
        N_bit = int(math.log2(N))
        poly = self.orderReverse(poly, N_bit)
        current = poly.clone()

        for i in range(N_bit):
            s = 2 ** i
            m = N // (2 * s)

            k = torch.arange(s, device=poly.device)
            exponents = k * (N // (2 * s))
            w_P = self.modExponent(w, exponents, M)

            w_P = w_P.repeat(m).to("cuda")

            current = current.view(m, 2 * s).to("cuda")
            even = current[:, :s]
            odd = current[:, s:]

            odd = (odd * w_P.view(-1, s)) % M

            upper = (even + odd) % M
            lower = (even - odd) % M

            current = torch.cat([upper, lower], dim=1).view(-1)

        return current

    def intt(self, points, M, N, w):
        inv_w = self.modInv(w, M)
        inv_N = self.modInv(N, M)
        poly = self.ntt(points, M, N, inv_w)
        return (poly * inv_N) % M

@profile
def negacyclic_convolution_test():
    ntt = NTT_GPU()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    M = 12289
    A = torch.arange(1, 1025, dtype=torch.int64, device=device)
    B = torch.arange(5, 1029, dtype=torch.int64, device=device)
    n = len(A)
    size = 2 * n
    A_ext = torch.cat([A, torch.zeros(size - n, dtype=torch.int64, device=device)])
    B_ext = torch.cat([B, torch.zeros(size - n, dtype=torch.int64, device=device)])

    w_2n = ntt.NthRootOfUnity(M, size)
    print(w_2n, size)

    time1 = time.time()
    A_ntt = ntt.ntt(A_ext, M, size, w_2n)
    B_ntt = ntt.ntt(B_ext, M, size, w_2n)
    time2 = time.time()
    print("NTT_A:", A_ntt)
    print("NTT_B:", B_ntt)
    print("NTT time: ", round(time2 - time1, 6))

    time3 = time.time()
    AB_ntt = (A_ntt * B_ntt) % M

    AB_time_2n = ntt.intt(AB_ntt, M, size, w_2n)

    C_negacyclic = (AB_time_2n[:n] - AB_time_2n[n:]) % M
    time4 = time.time()
    print("INTT time: ", round(time4 - time3, 6))

    print(f"\n--- Final result mod (x^n+1) ---")
    print("Negacyclic convolution result:", C_negacyclic.cpu().numpy())


if __name__ == "__main__":
    negacyclic_convolution_test()