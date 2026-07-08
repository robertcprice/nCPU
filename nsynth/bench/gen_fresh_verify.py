import json, math
from math import gcd, prod
def is_prime(x):
    if x<2: return False
    d=2
    while d*d<=x:
        if x%d==0: return False
        d+=1
    return True
def kth_prime(k):
    c=0;m=1
    while True:
        m+=1
        if is_prime(m):
            c+=1
            if c==k: return m
def kth_composite(k):
    c=0;m=3
    while True:
        m+=1
        if not is_prime(m):
            c+=1
            if c==k: return m
def omega_mult(x):
    c=0;d=2
    while d*d<=x:
        while x%d==0: x//=d; c+=1
        d+=1
    if x>1: c+=1
    return c
def proper_div_sum(n): return sum(d for d in range(1,n) if n%d==0) if n>=1 else 0
def divcount(n): return sum(1 for d in range(1,n+1) if n%d==0)
def is_pal(x): s=str(x); return s==s[::-1]
def collatz_orbit(n):
    o=[n]
    while n>1: n=n//2 if n%2==0 else 3*n+1; o.append(n)
    return o
def popcount(x): return bin(x).count('1')
def prime_after(n):
    m=n+1
    while not is_prime(m): m+=1
    return m
def next_pow2(n):
    p=1
    while p<n: p*=2
    return p
def nth_where(pred,n):
    c=0;m=0
    while True:
        m+=1
        if pred(m):
            c+=1
            if c==n: return m

# single-arg tasks: (fn, fresh inputs)
S = {
 "nth_prime": (kth_prime,[9,10,11,12,13,14,15,16,20,25]),
 "prime_after": (prime_after,[30,40,53,60,77,90,101,150,200,3]),
 "nth_composite": (kth_composite,[9,10,11,12,13,15,20,25,30,40]),
 "nth_palindrome": (lambda n:nth_where(is_pal,n),[20,25,30,40,50,13,14,16,18,60]),
 "nth_abundant": (lambda n:nth_where(lambda x:proper_div_sum(x)>x,n),[9,10,11,12,15,13,14,16,18,20]),
 "nth_squarefree": (lambda n:nth_where(lambda x:all(x%(d*d) for d in range(2,int(x**.5)+1)),n),[8,9,11,12,13,15,20,25,30,40]),
 "collatz_sum": (lambda n:sum(collatz_orbit(n)),[9,10,11,12,13,15,20,25,27,33]),
 "collatz_max": (lambda n:max(collatz_orbit(n)),[9,10,11,12,13,15,20,25,27,33]),
 "digit_factorial_sum": (lambda n:sum(math.factorial(int(d)) for d in str(n)),[11,22,34,56,78,90,123,405,169,222]),
 "nth_twin_prime_lower": (lambda n:[m for m in range(2,100000) if is_prime(m) and is_prime(m+2)][n-1],[9,10,11,12,13,15,20,25,30,40]),
 "count_prime_factors_with_mult": (omega_mult,[18,24,32,48,64,72,96,128,144,200]),
 "count_odd_divisors": (lambda n:sum(1 for d in range(1,n+1) if n%d==0 and d%2==1),[18,24,30,36,48,60,72,90,144,200]),
 "nth_three_divisor_number": (lambda n:nth_where(lambda x:divcount(x)==3,n),[9,10,11,12,13,14,15,16,18,20]),
 "sum_squares_of_divisors": (lambda n:sum(d*d for d in range(1,n+1) if n%d==0),[18,24,30,36,48,60,72,90,144,200]),
 "sum_first_n_primes_squared": (lambda n:sum(kth_prime(k)**2 for k in range(1,n+1)),[9,10,11,12,13,14,15,16,18,20]),
 "euler_totient": (lambda n:sum(1 for k in range(1,n+1) if gcd(k,n)==1),[18,24,30,36,48,60,72,90,100,200]),
 "nth_harshad": (lambda n:nth_where(lambda x:x%sum(int(d) for d in str(x))==0,n),[15,16,18,20,25,30,40,50,60,80]),
 "nth_odious": (lambda n:[x for x in range(0,100000) if popcount(x)%2==1][n-1],[9,10,11,12,13,15,20,25,30,40]),
 "nth_number_coprime_30": (lambda n:[x for x in range(1,100000) if gcd(x,30)==1][n-1],[9,10,11,12,13,15,20,25,30,40]),
 "digit_sum_of_square": (lambda n:sum(int(d) for d in str(n*n)),[11,13,17,19,23,29,31,37,41,50]),
 "next_power_of_two": (next_pow2,[10,15,20,33,63,65,100,200,500,1000]),
 "largest_proper_divisor": (lambda n:next(n//d for d in range(2,n+1) if n%d==0) if n>1 else 1,[18,24,30,36,48,60,72,91,100,200]),
 "count_prime_digits": (lambda n:sum(1 for c in str(n) if int(c) in(2,3,5,7)),[2357,1234,8888,9990,2233,5577,1010,4646,2727,3535]),
 "nth_prime_squared": (lambda n:kth_prime(n)**2,[9,10,11,12,13,14,15,16,18,20]),
 "count_primes_up_to": (lambda n:sum(1 for k in range(2,n+1) if is_prime(k)),[15,25,35,45,55,65,75,85,90,120]),
 "sum_first_n_factorials": (lambda n:sum(math.factorial(k) for k in range(1,n+1)),[9,10,11,12,13,6,7,8,5,4]),
 "sum_first_n_powers_of_2": (lambda n:2**(n+1)-2,[9,10,11,12,13,14,15,16,18,20]),
 "count_composites_up_to": (lambda n:sum(1 for k in range(4,n+1) if not is_prime(k)),[15,25,35,45,55,65,75,85,100,120]),
 "sum_first_n_primes_cubed": (lambda n:sum(kth_prime(k)**3 for k in range(1,n+1)),[9,10,11,12,13,14,15,6,7,5]),
 "product_first_n_odd": (lambda n:prod(2*k-1 for k in range(1,n+1)),[9,10,11,12,13,6,7,8,5,4]),
 "sum_first_n_hexagonal": (lambda n:sum(k*(2*k-1) for k in range(1,n+1)),[9,10,11,12,13,14,15,16,18,20]),
 "nth_non_prime": (lambda n:[x for x in range(1,100000) if not is_prime(x)][n-1],[9,10,11,12,13,15,20,25,30,40]),
 "sum_digits_nth_prime": (lambda n:sum(int(d) for d in str(kth_prime(n))),[9,10,11,12,13,14,15,16,18,20]),
 "count_semiprimes_up_to": (lambda n:sum(1 for k in range(2,n+1) if omega_mult(k)==2),[12,18,24,35,40,50,60,75,90,100]),
 "nth_prime_minus_index": (lambda n:kth_prime(n)-n,[9,10,11,12,13,14,15,16,18,20]),
 "nth_even_semiprime": (lambda n:[m for m in range(2,100000) if m%2==0 and is_prime(m//2)][n-1],[9,10,11,12,13,14,15,16,18,20]),
 "nth_prime_cubed": (lambda n:kth_prime(n)**3,[9,10,11,12,13,14,7,6,5,8]),
 "count_primes_even_digit_sum": (lambda n:sum(1 for k in range(2,n+1) if is_prime(k) and sum(int(d) for d in str(k))%2==0),[15,25,35,45,55,65,75,85,90,120]),
 "nth_prime_mod_10": (lambda n:kth_prime(n)%10,[9,10,11,12,13,14,15,16,18,20]),
 "count_primes_ending_in_1": (lambda n:sum(1 for k in range(2,n+1) if is_prime(k) and k%10==1),[15,32,42,52,62,72,82,92,110,150]),
 "nth_prime_squared_minus_1": (lambda n:kth_prime(n)**2-1,[9,10,11,12,13,14,15,16,18,20]),
 "sum_first_n_primes_doubled": (lambda n:sum(2*kth_prime(k) for k in range(1,n+1)),[9,10,11,12,13,14,15,16,18,20]),
 "nth_composite_squared": (lambda n:kth_composite(n)**2,[9,10,11,12,13,14,15,16,18,20]),
 "count_odd_composites_up_to": (lambda n:sum(1 for k in range(9,n+1,2) if not is_prime(k)),[15,25,35,45,55,65,75,85,100,120]),
 "nth_prime_plus_n": (lambda n:kth_prime(n)+n,[9,10,11,12,13,14,15,16,18,20]),
 "sum_first_n_composites_squared": (lambda n:sum(kth_composite(k)**2 for k in range(1,n+1)),[9,10,11,12,13,6,7,8,5,4]),
}
T2 = {
 "count_primes_in_range": (lambda a,b:sum(1 for k in range(a,b+1) if is_prime(k)),[[1,20],[10,30],[5,15],[20,40],[1,50],[30,50],[40,60],[15,25],[11,13],[2,7]]),
 "sum_multiples_of_k_below": (lambda k,n:sum(x for x in range(k,n,k)),[[3,20],[4,30],[6,40],[2,15],[5,50],[7,70],[9,90],[3,100],[8,64],[10,55]]),
 "sum_of_range": (lambda a,b:sum(range(a,b+1)),[[1,20],[5,15],[10,30],[0,9],[7,14],[20,40],[50,55],[100,110],[11,19],[3,3]]),
 "sum_squares_in_range": (lambda a,b:sum(k*k for k in range(a,b+1)),[[1,6],[2,7],[3,8],[1,10],[4,9],[5,12],[0,5],[6,11],[10,15],[2,20]]),
 "count_odd_in_range": (lambda a,b:sum(1 for k in range(a,b+1) if k%2==1),[[1,20],[2,18],[5,25],[0,15],[10,40],[3,33],[7,17],[11,29],[50,60],[4,44]]),
 "max_prime_in_range": (lambda a,b:max((k for k in range(a,b+1) if is_prime(k)),default=0),[[1,20],[10,30],[14,25],[1,50],[30,45],[50,70],[90,100],[40,60],[2,9],[5,5]]),
 "sum_evens_in_range": (lambda a,b:sum(k for k in range(a,b+1) if k%2==0),[[1,20],[2,18],[0,15],[3,25],[10,40],[5,15],[7,21],[11,31],[50,60],[4,44]]),
 "sum_cubes_in_range": (lambda a,b:sum(k**3 for k in range(a,b+1)),[[1,4],[2,5],[3,6],[1,7],[4,8],[0,4],[5,9],[6,10],[2,3],[1,6]]),
 "count_squares_in_range": (lambda a,b:sum(1 for k in range(0,int(b**.5)+2) if a<=k*k<=b),[[1,20],[4,36],[2,50],[1,100],[10,90],[26,64],[5,80],[1,200],[50,150],[3,8]]),
 "min_prime_in_range": (lambda a,b:next((k for k in range(a,b+1) if is_prime(k)),0),[[1,20],[8,30],[14,25],[24,40],[90,100],[50,60],[6,20],[100,120],[4,9],[1,2]]),
 "count_composites_in_range": (lambda a,b:sum(1 for k in range(a,b+1) if k>=4 and not is_prime(k)),[[1,20],[4,25],[2,10],[10,40],[15,35],[50,60],[5,15],[8,18],[30,50],[4,4]]),
 "count_evens_in_range": (lambda a,b:sum(1 for k in range(a,b+1) if k%2==0),[[1,20],[2,18],[0,15],[3,25],[10,40],[5,15],[7,21],[11,31],[50,60],[4,44]]),
}
lines=[]
for n,(fn,inp) in {**S}.items():
    try:
        exs=[{"in":[i],"out":fn(i)} for i in inp]
        lines.append(json.dumps({"name":n+"_FRESH","examples":exs}))
    except Exception as e: print("skip",n,e)
for n,(fn,inp) in T2.items():
    try:
        exs=[{"in":list(a),"out":fn(*a)} for a in inp]
        lines.append(json.dumps({"name":n+"_FRESH","examples":exs}))
    except Exception as e: print("skip",n,e)
open("/private/tmp/fresh_tasks.jsonl","w").write("\n".join(lines)+"\n")
print("emitted fresh-input tasks:",len(lines))
