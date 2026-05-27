pub(super) fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let tmp = b;
        b = a % b;
        a = tmp;
    }
    a
}

pub(super) fn fibonacci(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let next = a + b;
        a = b;
        b = next;
    }
    a
}

pub(super) fn digit_sum(mut n: i64) -> i64 {
    n = n.abs();
    let mut total = 0;
    while n > 0 {
        total += n % 10;
        n /= 10;
    }
    total
}

pub(super) fn reverse_digits(mut n: i64) -> i64 {
    n = n.abs();
    let mut acc = 0;
    while n > 0 {
        acc = (acc * 10) + (n % 10);
        n /= 10;
    }
    acc
}

pub(super) fn digit_count(mut n: i64) -> i64 {
    n = n.abs();
    if n == 0 {
        return 1;
    }
    let mut acc = 0;
    while n > 0 {
        acc += 1;
        n /= 10;
    }
    acc
}

pub(super) fn count_even_digits(mut n: i64) -> i64 {
    n = n.abs();
    if n == 0 {
        return 1;
    }
    let mut acc = 0;
    while n > 0 {
        if (n % 10) % 2 == 0 {
            acc += 1;
        }
        n /= 10;
    }
    acc
}

pub(super) fn collatz_steps(mut n: i64) -> i64 {
    let mut steps = 0;
    while n > 1 {
        if n % 2 == 0 {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        steps += 1;
    }
    steps
}

pub(super) fn is_prime(n: i64) -> i64 {
    if n < 2 {
        return 0;
    }
    if n == 2 {
        return 1;
    }
    if n % 2 == 0 {
        return 0;
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return 0;
        }
        i += 2;
    }
    1
}

pub(super) fn count_words(s: &str) -> i64 {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return 0;
    }
    trimmed.split(' ').filter(|part| !part.is_empty()).count() as i64
}

pub(super) fn euler_totient(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut result = n;
    let mut p = 2;
    let mut temp = n;
    while p * p <= temp {
        if temp % p == 0 {
            while temp % p == 0 {
                temp /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if temp > 1 {
        result -= result / temp;
    }
    result
}

pub(super) fn triangular_check(n: i64) -> i64 {
    let mut k = 0;
    while k * (k + 1) / 2 <= n {
        if k * (k + 1) / 2 == n {
            return 1;
        }
        k += 1;
    }
    0
}

pub(super) fn harmonic_sum(n: i64) -> i64 {
    let mut total = 0;
    let mut i = 1;
    while i <= n {
        total += 1000 / i;
        i += 1;
    }
    total
}

pub(super) fn second_max(arr: &[i64]) -> i64 {
    let mut first = arr[0];
    let mut second = arr[0];
    for &item in arr {
        if item > first {
            second = first;
            first = item;
        } else if item > second {
            second = item;
        }
    }
    second
}

pub(super) fn array_range(arr: &[i64]) -> i64 {
    let lo = *arr.iter().min().unwrap();
    let hi = *arr.iter().max().unwrap();
    hi - lo
}

pub(super) fn sum_of_divisors(n: i64) -> i64 {
    (1..=n).filter(|d| n % d == 0).sum()
}

pub(super) fn sum_odd_digits(mut n: i64) -> i64 {
    let mut acc = 0;
    while n > 0 {
        let d = n % 10;
        if d % 2 == 1 {
            acc += d;
        }
        n /= 10;
    }
    acc
}
