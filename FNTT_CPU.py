#fast-ntt
import random
import time
from memory_profiler import profile
from sympy import isprime

class NTT:

    def isInteger(self, M):
        return type(M).__name__ == 'int'

    def isPrime(self, M):
        assert self.isInteger(M), 'Not an integer.'
        return isprime(M)

    def modExponent(self, base, power, M):
        result = 1
        power = int(power)
        base %= M
        while power > 0:
            if power & 1:
                result = (result * base) % M
            base = (base * base) % M
            power >>= 1
        return result

    def modInv(self, x, M):
        t, new_t = 0, 1
        r, new_r = M, x
        while new_r != 0:
            quotient = r // new_r
            t, new_t = new_t, (t - quotient * new_t)
            r, new_r = new_r, (r % new_r)
        if r > 1:
            return "x is not invertible."
        if t < 0:
            t += M
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

    def bitReverse(self, num, length_):
        rev_num = 0
        for i in range(length_):
            if (num >> i) & 1:
                rev_num |= 1 << (length_ - 1 - i)
        return rev_num

    def orderReverse(self, poly, N_bit):

        for i, coeff in enumerate(poly):
            rev_i = self.bitReverse(i, N_bit)
            if rev_i > i:
                coeff ^= poly[rev_i]
                poly[rev_i] ^= coeff
                coeff ^= poly[rev_i]
                poly[i] = coeff
        return poly

    def ntt(self, poly, M, N, w):

        N_bit = N.bit_length() - 1
        self.orderReverse(poly, N_bit)
        for i in range(N_bit):
            points1, points2 = [], []
            for j in range(0, N // 2):
                shift_bits = N_bit - 1 - i
                P = (j >> shift_bits) << shift_bits
                w_P = self.modExponent(w, P, M)
                odd = poly[2*j + 1] * w_P
                even = poly[2*j]
                points1.append((even + odd) % M)
                points2.append((even - odd) % M)
                points = points1 + points2
            if i != N_bit:
                poly = points
        return points

    def intt(self, points, M, N, w):
        inv_w = self.modInv(w, M)
        inv_N = self.modInv(N, M)
        poly = self.ntt(points, M, N, inv_w)
        for i in range(N):
            poly[i] = (poly[i] * inv_N) % M
        return poly

@profile
def negacyclic_convolution_test():

    ntt = NTT()
    M = 12289
    A = list(range(1, 1025))
    B = list(range(5, 1029))
    n = len(A)

    print(f"Modulus = {M}")
    print("Polynomial A =", A)
    print("Polynomial B =", B)
    print(f"Target ring: Z_{M}[x]/(x^{n}+1)  (negacyclic)")

    size = 2 * n
    A_ext = A + [0]*(size - n)
    B_ext = B + [0]*(size - n)

    w_2n = ntt.NthRootOfUnity(M, size)
    print(f"\n2n-th root of unity = {w_2n},  2n={size}")

    wn = ntt.modExponent(w_2n, n, M)
    print(f"w_2n^{n} mod M = {wn} (expect -1 for negacyclic)")
    a = time.time()
    A_ntt = ntt.ntt(A_ext[:], M, size, w_2n)
    B_ntt = ntt.ntt(B_ext[:], M, size, w_2n)
    b = time.time()
    print("NTT time", round(b-a, 6))

    c = time.time()
    AB_ntt = [(A_ntt[i] * B_ntt[i]) % M for i in range(size)]

    AB_time_2n = ntt.intt(AB_ntt, M, size, w_2n)

    C_negacyclic = [(AB_time_2n[i] - AB_time_2n[i+n]) % M for i in range(n)]
    d = time.time()
    print("INTT time", round(d-c, 6))
    print("A_ntt =", A_ntt)
    print("B_ntt =", B_ntt)
    print("AB_ntt =", AB_ntt)
    print("AB_intt_2n =", AB_time_2n)
    print("C_negacyclic =", C_negacyclic)

if __name__=="__main__":
    negacyclic_convolution_test()
