use super::*;

#[test]
fn search_only_generalizes_on_string_holdout_cases() {
    assert_search_generalizes(
        "trimmed_len_v0",
        vec![
            (vec![Value::Str("   hi there   ".to_string())], 8),
            (vec![Value::Str("      ".to_string())], 0),
        ],
    );
    assert_search_generalizes(
        "vowel_count_v0",
        vec![
            (vec![Value::Str("queue".to_string())], 4),
            (vec![Value::Str("sky".to_string())], 0),
        ],
    );
    assert_search_generalizes(
        "contains_cat_v0",
        vec![
            (vec![Value::Str("bobcat".to_string())], 1),
            (vec![Value::Str("atlas".to_string())], 0),
        ],
    );
    assert_search_generalizes(
        "starts_with_m_v0",
        vec![
            (vec![Value::Str("m".to_string())], 1),
            (vec![Value::Str("Map".to_string())], 0),
            (vec![Value::Str("moss".to_string())], 1),
        ],
    );
    assert_search_generalizes(
        "palindrome_check_v0",
        vec![
            (vec![Value::Str("abba".to_string())], 1),
            (vec![Value::Str("abca".to_string())], 0),
        ],
    );
    assert_search_generalizes(
        "count_words_v0",
        vec![
            (vec![Value::Str("  many   spaces here  ".to_string())], 3),
            (vec![Value::Str("single".to_string())], 1),
        ],
    );
}
