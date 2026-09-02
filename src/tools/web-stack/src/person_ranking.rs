//! Shared person-role classification, ranking, and validation helpers.

use serde_json::Value;
use std::cmp::Ordering;

pub const DEFAULT_PERSON_PRIORITIES: &[&str] = &[
    "Geschäftsführung/Gesamtverantwortung",
    "Prokura",
    "Leitung Finanzen",
    "Einkauf",
    "Supply Chain Management",
    "Operations",
    "Technik",
    "Entwicklung",
];

const ROLE_KEYWORD_GROUPS: &[&[&str]] = &[
    &[
        "geschäftsführ",
        "geschaeftsfuehr",
        "ceo",
        "managing director",
        "inhaber",
        "vorstand",
    ],
    &["prokur"],
    &["cfo", "finanz", "controlling"],
    &["einkauf", "procurement", "purchasing"],
    &["supply chain", "logistik", "scm"],
    &[
        "operations",
        "coo",
        "werkleit",
        "betriebsleit",
        "produktionsleit",
    ],
    &[
        "technik",
        "cto",
        "technische leitung",
        "technischer leiter",
        "technische leiterin",
        "instandhaltung",
    ],
    &["entwicklung", "r&d", "f&e", "research"],
];

pub fn role_category(role: &str) -> usize {
    let normalized = role.trim().to_lowercase();
    ROLE_KEYWORD_GROUPS
        .iter()
        .position(|keywords| normalized_contains_any(&normalized, keywords))
        .map(|index| index + 1)
        .unwrap_or(99)
}

pub fn contains_role_keyword(value: &str) -> bool {
    let normalized = value.trim().to_lowercase();
    role_category(value) != 99
        || normalized_contains_any(&normalized, GENERIC_ROLE_SUBSTRINGS)
        || normalized_words_contain_any(&normalized, GENERIC_ROLE_WORDS)
}

/// Generic role vocabulary that does not map to a priority category but still
/// marks a value as a job function rather than a place or a person name
/// (review finding N4: "Account Manager", "Verkauf", "HR" were rejected).
const GENERIC_ROLE_SUBSTRINGS: &[&str] = &[
    "leiter",
    "leiterin",
    "leitung",
    "head of",
    "manager",
    "director",
    "direktor",
    "vorstand",
    "assistenz",
    "assistant",
    "sales",
    "verkauf",
    "vertrieb",
    "marketing",
    "personal",
    "human resources",
    "informatik",
    "consultant",
    "berater",
    "referent",
    "sachbearbeit",
    "kaufmann",
    "kauffrau",
    "kaufmänn",
    "controller",
    "buchhalt",
    "ingenieur",
    "engineer",
    "specialist",
    "spezialist",
    "expert",
    "koordinator",
    "coordinator",
    "projekt",
    "product",
    "produkt",
    "partner",
    "gesellschafter",
    "owner",
    "founder",
    "gründer",
    "chef",
    "officer",
    "analyst",
    "architekt",
    "architect",
    "qualität",
    "quality",
    "service",
    "support",
    "recruit",
    "key account",
];

/// Short tokens that only count as whole words (substring matching would hit
/// "leiter" for "it" or "uhr" for "hr").
const GENERIC_ROLE_WORDS: &[&str] = &[
    "hr", "it", "ceo", "cfo", "coo", "cto", "cio", "cmo", "cso", "vp", "pm",
];

fn normalized_words_contain_any(value: &str, words: &[&str]) -> bool {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .any(|word| words.contains(&word))
}

pub fn priority_rank(role: &str, priorities: &[String]) -> usize {
    let category = role_category(role);
    if category == 99 {
        return 99;
    }
    let defaults = DEFAULT_PERSON_PRIORITIES
        .iter()
        .map(|value| (*value).to_string())
        .collect::<Vec<_>>();
    let ordered = if priorities.is_empty() {
        defaults.as_slice()
    } else {
        priorities
    };
    ordered
        .iter()
        .position(|label| priority_label_category(label) == category)
        .map(|index| index + 1)
        .unwrap_or(99)
}

pub fn compare_person_records(left: &Value, right: &Value, priorities: &[String]) -> Ordering {
    let left_role = record_role(left);
    let right_role = record_role(right);
    priority_rank(&left_role, priorities)
        .cmp(&priority_rank(&right_role, priorities))
        .then_with(|| evidence_count(right).cmp(&evidence_count(left)))
        .then_with(|| person_name(left).cmp(&person_name(right)))
}

pub fn role_is_valid(value: &str, person_name: &str, locations: &[String]) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return false;
    }
    if contains_role_keyword(value) {
        return true;
    }
    let normalized = normalize_comparison(value);
    if normalized.is_empty()
        || normalized == normalize_comparison(person_name)
        || locations
            .iter()
            .any(|location| normalized == normalize_comparison(location))
        || matches!(normalized.as_str(), "dr" | "prof" | "professor")
    {
        return false;
    }
    !is_short_capitalized_name_or_place(value)
}

fn priority_label_category(label: &str) -> usize {
    let normalized = label.trim().to_lowercase();
    if normalized_contains_any(
        &normalized,
        &[
            "geschäftsführ",
            "geschaeftsfuehr",
            "gesamtverantwort",
            "ceo",
            "vorstand",
            "inhaber",
        ],
    ) {
        1
    } else if normalized.contains("prokur") {
        2
    } else if normalized_contains_any(&normalized, &["finanz", "cfo", "controlling"]) {
        3
    } else if normalized_contains_any(&normalized, &["einkauf", "procurement", "purchasing"]) {
        4
    } else if normalized_contains_any(&normalized, &["supply chain", "logistik", "scm"]) {
        5
    } else if normalized_contains_any(
        &normalized,
        &["operations", "coo", "produktion", "werk", "betrieb"],
    ) {
        6
    } else if normalized_contains_any(&normalized, &["technik", "cto", "instandhaltung"]) {
        7
    } else if normalized_contains_any(&normalized, &["entwicklung", "r&d", "f&e", "research"]) {
        8
    } else {
        99
    }
}

fn normalized_contains_any(value: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|keyword| value.contains(keyword))
}

fn is_short_capitalized_name_or_place(value: &str) -> bool {
    let words = value.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() || words.len() > 3 {
        return false;
    }
    words.iter().all(|word| {
        !word.is_empty()
            && word.chars().all(char::is_alphabetic)
            && word.chars().next().is_some_and(char::is_uppercase)
    })
}

fn normalize_comparison(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

fn record_role(record: &Value) -> String {
    ["person_funktion", "person_position", "role", "position"]
        .iter()
        .filter_map(|key| {
            record
                .get(*key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn evidence_count(record: &Value) -> u64 {
    record
        .get("evidence_count")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

fn person_name(record: &Value) -> String {
    let first = record
        .get("person_vorname")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let last = record
        .get("person_nachname")
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!("{first} {last}").trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn person_priority_order_is_category_then_evidence_then_name() {
        let priorities = vec![
            "Geschäftsführung/Gesamtverantwortung".to_string(),
            "Prokura".to_string(),
            "Leitung Finanzen".to_string(),
            "Einkauf".to_string(),
            "Supply Chain Management".to_string(),
            "Operations".to_string(),
            "Technik".to_string(),
            "Entwicklung".to_string(),
        ];
        let mut records = vec![
            json!({"person_vorname":"Eva","person_nachname":"Dev","person_funktion":"Entwicklungsleiter","evidence_count":9}),
            json!({"person_vorname":"Paula","person_nachname":"Prokura","person_funktion":"Prokurist","evidence_count":2}),
            json!({"person_vorname":"Greta","person_nachname":"Chef","person_funktion":"Geschäftsführerin","evidence_count":1}),
        ];
        records.sort_by(|left, right| compare_person_records(left, right, &priorities));
        assert_eq!(records[0]["person_vorname"], "Greta");
        assert_eq!(records[1]["person_vorname"], "Paula");
        assert_eq!(records[2]["person_vorname"], "Eva");
    }

    #[test]
    fn role_validation_rejects_names_places_and_titles() {
        for (value, expected) in [
            ("Leipzig", false),
            ("Tim Nils Berner", false),
            ("Geschäftsführer", true),
            ("Head of Supply Chain", true),
            ("Leiterin Einkauf", true),
            ("Dr.", false),
            ("Account Manager", true),
            ("Verkauf", true),
            ("HR", true),
            ("IT", true),
            ("Sales", true),
            ("Werkleiter Leipzig", true),
            ("Berlin", false),
        ] {
            assert_eq!(
                role_is_valid(value, "Tim Nils Berner", &[]),
                expected,
                "{value}"
            );
        }
    }
}
