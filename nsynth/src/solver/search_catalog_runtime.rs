pub(super) fn min_consecutive_sum(arr: &[i64]) -> i64 {
    if arr.is_empty() {
        return 0;
    }
    let mut current = 0i64;
    let mut best = arr[0];
    for &item in arr {
        current = if current < 0 { current + item } else { item };
        best = best.min(current);
    }
    best
}

pub(super) fn kth_smallest_rust(arr: &[i64], k: i64) -> i64 {
    if k < 1 || k as usize > arr.len() {
        return i64::MIN;
    }
    let mut values = arr.to_vec();
    values.sort();
    values[(k - 1) as usize]
}

pub(super) fn max_stock_profit_rust(prices: &[i64]) -> i64 {
    let mut min_price = prices[0];
    let mut best = 0i64;
    for &price in prices {
        if price < min_price {
            min_price = price;
        }
        let profit = price - min_price;
        if profit > best {
            best = profit;
        }
    }
    best
}

pub(super) fn is_sorted_rust(arr: &[i64]) -> i64 {
    if arr.windows(2).all(|window| window[0] <= window[1]) {
        1
    } else {
        0
    }
}

pub(super) fn strictly_increasing_rust(arr: &[i64]) -> i64 {
    if arr.windows(2).all(|window| window[0] < window[1]) {
        1
    } else {
        0
    }
}

pub(super) fn has_strictly_increasing_run_rust(arr: &[i64], length: i64) -> i64 {
    if length <= 1 || arr.is_empty() {
        return 1;
    }
    let mut run = 1i64;
    for index in 1..arr.len() {
        if arr[index] > arr[index - 1] {
            run += 1;
            if run >= length {
                return 1;
            }
        } else {
            run = 1;
        }
    }
    0
}

pub(super) fn first_index_of_rust(arr: &[i64], target: i64) -> i64 {
    for (i, &v) in arr.iter().enumerate() {
        if v == target {
            return i as i64;
        }
    }
    -1
}

pub(super) fn longest_increasing_run_rust(arr: &[i64]) -> i64 {
    let mut best = 1i64;
    let mut current = 1i64;
    for index in 1..arr.len() {
        if arr[index] > arr[index - 1] {
            current += 1;
            if current > best {
                best = current;
            }
        } else {
            current = 1;
        }
    }
    best
}

pub(super) fn digital_root_rust(mut n: i64) -> i64 {
    while n >= 10 {
        let mut sum = 0i64;
        while n > 0 {
            sum += n % 10;
            n /= 10;
        }
        n = sum;
    }
    n
}

pub(super) fn two_sum_exists_rust(arr: &[i64], target: i64) -> i64 {
    for i in 0..arr.len() {
        for j in (i + 1)..arr.len() {
            if arr[i] + arr[j] == target {
                return 1;
            }
        }
    }
    0
}

pub(super) fn count_distinct_rust(arr: &[i64]) -> i64 {
    let mut values = arr.to_vec();
    values.sort();
    values.dedup();
    values.len() as i64
}

pub(super) fn binary_search_rust(arr: &[i64], target: i64) -> i64 {
    let mut lo = 0i64;
    let mut hi = arr.len() as i64 - 1;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        if arr[mid as usize] == target {
            return mid;
        }
        if arr[mid as usize] < target {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    -1
}

pub(super) fn longest_plateau_rust(arr: &[i64]) -> i64 {
    let mut best = 1i64;
    let mut current = 1i64;
    for index in 1..arr.len() {
        if arr[index] == arr[index - 1] {
            current += 1;
            if current > best {
                best = current;
            }
        } else {
            current = 1;
        }
    }
    best
}

pub(super) fn prefix_max_sum_rust(arr: &[i64]) -> i64 {
    let mut running_max = arr[0];
    let mut total = 0i64;
    for &value in arr {
        if value > running_max {
            running_max = value;
        }
        total += running_max;
    }
    total
}

pub(super) fn kth_from_end_rust(arr: &[i64], k: i64) -> i64 {
    if k < 1 || k as usize > arr.len() {
        return i64::MIN;
    }
    arr[arr.len() - k as usize]
}

pub(super) fn max_consecutive_sum(arr: &[i64]) -> i64 {
    if arr.is_empty() {
        return 0;
    }
    let mut current = 0i64;
    let mut best = arr[0];
    for &item in arr {
        current = if current > 0 { current + item } else { item };
        best = best.max(current);
    }
    best
}
