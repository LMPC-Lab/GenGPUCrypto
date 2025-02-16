# NTT based on PyTorch
import random
from sympy import isprime
import time
import torch
from memory_profiler import profile

class NTT:
    def isInteger(self, M):
        return type(M).__name__ == 'int'

    def isPrime(self, M):
        assert self.isInteger(M), 'Not an integer.'
        return isprime(M)

    def modExponent(self, base, power, M):
        result = 1
        power = int(power)
        base = base % M
        while power > 0:
            if power & 1:
                result = (result * base) % M
            base = (base * base) % M
            power = power >> 1
        return result

    def modInv(self, x, M):
        t, new_t, r, new_r = 0, 1, M, x
        while new_r != 0:
            quotient = int(r / new_r)
            t, new_t = new_t, (t - quotient * new_t)
            r, new_r = new_r, (r % new_r)
        if r > 1:
            return "x is not invertible."
        if t < 0:
            t = t + M
        return t

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

    def isNthRootOfUnity(self, M, N, beta):
        return self.modExponent(beta, N, M) == 1

    def primitive2NthRootOfUnity(self, M, N):
        assert self.isPrime(M), "M must be a prime number"
        assert (M - 1) % (2 * N) == 0, "2N must divide phi(M)"

        omega = self.NthRootOfUnity(M, N)
        for candidate in range(1, M):
            if (candidate * candidate) % M == omega:
                if self.modExponent(candidate, N, M) == M - 1:
                    return candidate
        return None

    def mod_exp(self, base, exponent, modulus):
        # prevent overflow
        result = 1
        base = base % modulus
        while exponent > 0:
            if exponent % 2 == 1:
                result = (result * base) % modulus
            exponent = exponent // 2
            base = (base * base) % modulus
        return result

    def construct_matrix_ntt(self, n, psi, q):
        i = torch.arange(n, dtype=torch.int64).reshape(n, 1)
        j = torch.arange(n, dtype=torch.int64).reshape(1, n)
        exponents = (2 * i * j + i).T
        matrix = torch.zeros((n, n), dtype=torch.int64)
        for row in range(n):
            for col in range(n-1, -1, -1):
                exp = exponents[row, col].item()
                matrix[row, col] = self.mod_exp(psi, exp, q)
        return matrix.to("cuda")

    def construct_matrix_intt(self, n, psi, q):
        i = torch.arange(n, dtype=torch.int64).reshape(n, 1)
        j = torch.arange(n, dtype=torch.int64).reshape(1, n)
        psi_inv = self.modInv(psi, q)
        exponents = 2 * i * j + j
        matrix = torch.zeros((n, n), dtype=torch.int64)
        for row in range(n):
            for col in range(n):
                exp = exponents[row, col].item()
                matrix[row, col] = self.mod_exp(psi_inv, exp, q)
        n_inv = self.modInv(n, q)
        matrix = (matrix * n_inv) % q
        return matrix.to("cuda")

    def ntt(self, poly, M, matrix_ntt):

        return torch.mv(matrix_ntt.to(torch.float64), poly.to(torch.float64)).to(torch.int64) % M

    def intt(self, M, ntt_a, ntt_b, matrix_intt):

        vector = ((ntt_a * ntt_b) % M).to("cuda")
        return torch.mv(matrix_intt.to(torch.float64), vector.to(torch.float64))

@profile
def Bench():
    ntt = NTT()
    M = 8380417
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    poly_a = torch.tensor(list(range(1, 1025)), dtype=torch.int64, device=device)
    poly_b = torch.tensor(list(range(5, 1029)), dtype=torch.int64, device=device)
    print("Modulus : %d" % M)
    print("Polynomial_a : ", poly_a)
    print("Polynomial_b : ", poly_b)
    N = len(poly_a)

    bnr = ntt.primitive2NthRootOfUnity(M, N)
    print("2N-th RootOfUnity: ", bnr)

    a = time.time()
    matrix_ntt = ntt.construct_matrix_ntt(N, bnr, M)
    matrix_intt = ntt.construct_matrix_intt(N, bnr, M).T
    b = time.time()
    print("Matrix Generation: ", round(b - a, 6))

    start_time_ntt = time.time()
    ntt_a = ntt.ntt(poly_a, M, matrix_ntt)
    ntt_b = ntt.ntt(poly_b, M, matrix_ntt)
    end_time_ntt = time.time()
    execution_time_ntt = end_time_ntt - start_time_ntt
    print("NTT time", round(execution_time_ntt, 6))
    print("NTT_a = ", ntt_a)
    print("NTT_b = ", ntt_b)

    start_time_intt = time.time()
    intt_poly = (ntt.intt(M, ntt_a, ntt_b, matrix_intt).to(torch.int64)) % M
    end_time_intt = time.time()
    execution_time_intt = end_time_intt - start_time_intt
    print(intt_poly)
    print("INTT time:", round(execution_time_intt, 6))

Bench()
