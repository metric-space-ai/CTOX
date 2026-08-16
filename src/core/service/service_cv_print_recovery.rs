// Origin: CTOX
// License: AGPL-3.0-only

use super::clip_text;

pub(super) fn cv_print_heading_matches(value: &str, heading: &str) -> bool {
    let value = cv_print_normalized_heading(value);
    let heading = cv_print_normalized_heading(heading);
    if value == heading {
        return true;
    }
    heading == "ausbildungsweg" && matches!(value.as_str(), "ausbildung" | "studium")
}

pub(super) fn cv_print_normalized_heading(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| matches!(ch, ':' | ';' | '.'))
        .to_ascii_lowercase()
}

pub(super) fn cv_print_clean_line(value: &str) -> String {
    value
        .trim()
        .trim_start_matches(|ch: char| matches!(ch, '•' | '-' | '–' | '*' | '·'))
        .trim()
        .to_string()
}

pub(super) fn cv_print_clean_token(value: &str) -> String {
    value
        .trim_matches(|ch: char| matches!(ch, ',' | ';' | '.' | '(' | ')' | '[' | ']' | ':' | '|'))
        .to_string()
}

pub(super) fn cv_print_is_month_token(value: &str) -> bool {
    let lower = cv_print_clean_token(value).to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "januar"
            | "jan"
            | "februar"
            | "feb"
            | "märz"
            | "maerz"
            | "mrz"
            | "april"
            | "apr"
            | "mai"
            | "juni"
            | "jun"
            | "juli"
            | "jul"
            | "august"
            | "aug"
            | "september"
            | "sep"
            | "oktober"
            | "okt"
            | "oct"
            | "november"
            | "nov"
            | "dezember"
            | "dez"
            | "dec"
    )
}

pub(super) fn cv_print_split_inline_title_org(raw_title: &str) -> (String, String) {
    let cleaned = cv_print_clean_line(raw_title)
        .trim_matches('|')
        .trim()
        .to_string();
    if let Some((title, org)) = cleaned.split_once(" | ") {
        return (
            clip_text(title.trim(), 100),
            clip_text(org.trim().trim_matches('.'), 100),
        );
    }
    if let Some((title, org)) = cleaned.split_once(" bei ") {
        return (
            clip_text(title.trim(), 100),
            clip_text(org.trim().trim_matches('.'), 100),
        );
    }
    (clip_text(cleaned.trim(), 100), String::new())
}

pub(super) fn cv_print_line_is_entry_detail(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    value.starts_with('•')
        || lower.starts_with("projekt")
        || lower.starts_with("projekte")
        || lower.starts_with("aufgaben")
        || lower.starts_with("erfolge")
        || lower.starts_with("thesis")
        || lower.starts_with("training")
        || lower.starts_with("aktivitäten")
        || lower.starts_with("aktivitaeten")
}

pub(super) fn cv_print_line_is_detail_label(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    matches!(
        lower.trim_end_matches(':'),
        "aufgaben" | "aufgaben / erfolge" | "erfolge" | "projekte" | "projekt"
    )
}

pub(super) fn cv_print_clean_detail_line(value: &str) -> String {
    let cleaned = cv_print_clean_line(value);
    if let Some((label, rest)) = cleaned.split_once(':') {
        let label_lower = label.to_ascii_lowercase();
        if matches!(
            label_lower.as_str(),
            "projekt" | "projekte" | "thesis" | "training" | "aktivitäten" | "aktivitaeten"
        ) {
            return rest.trim().to_string();
        }
    }
    cleaned
}
