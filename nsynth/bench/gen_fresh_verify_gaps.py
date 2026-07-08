import json
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
def nth_even_fib(n):
    a,b=1,2;c=0
    while True:
        if b%2==0:
            c+=1
            if c==n: return b
        a,b=b,a+b
def sopfr(n):
    s=0;d=2;t=n
    while d*d<=t:
        while t%d==0: s+=d; t//=d
        d+=1
    if t>1: s+=t
    return s
def sum_first_n_fib(n):
    a,b,s=1,1,0
    for _ in range(n): s+=a; a,b=b,a+b
    return s
def nth_padovan(n):
    p=[1,1,1]
    if n<=3: return p[n-1]
    for _ in range(n-3): p=[p[1],p[2],p[0]+p[1]]
    return p[2]
def sum_first_n_primes(n):
    s=0;c=0;m=1
    while c<n:
        m+=1
        if is_prime(m): s+=m; c+=1
    return s
def sum_first_n_odd_primes(n):
    s=0;c=0;m=2
    while c<n:
        m+=1
        if is_prime(m) and m>2: s+=m; c+=1
    return s
def count_primes_up_to_double(n): return sum(1 for k in range(2,2*n+1) if is_prime(k))
def sum_first_n_primes_minus_n(n):
    s=0;c=0;m=1
    while c<n:
        m+=1
        if is_prime(m): c+=1; s+=m
    return s-n
S={
 "nth_even_fibonacci":(nth_even_fib,[9,10,11,12,13,14,15,16,17,18]),
 "sopfr":(sopfr,[18,24,32,48,60,72,96,100,128,144]),
 "sum_first_n_fibonacci":(sum_first_n_fib,[9,10,11,12,13,14,15,16,17,18]),
 "nth_padovan":(nth_padovan,[9,11,12,13,14,15,16,17,18,20]),
 "sum_first_n_primes":(sum_first_n_primes,[9,10,11,12,13,14,15,16,17,18]),
 "sum_first_n_odd_primes":(sum_first_n_odd_primes,[9,10,11,12,13,14,15,16,17,18]),
 "count_primes_up_to_double":(count_primes_up_to_double,[9,10,11,12,13,15,20,25,30,40]),
 "sum_first_n_primes_minus_n":(sum_first_n_primes_minus_n,[9,10,11,12,13,14,15,16,17,18]),
}
lines=[json.dumps({"name":n+"_FRESH","examples":[{"in":[i],"out":f(i)} for i in inp]}) for n,(f,inp) in S.items()]
open("/private/tmp/fresh_tasks3.jsonl","w").write("\n".join(lines)+"\n")
print("emitted gap fresh tasks:",len(lines))
