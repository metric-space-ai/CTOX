use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::OnceLock,
};

#[cfg(test)]
use anyhow::bail;
use anyhow::{Context, Result};
use regex::Regex;
use scraper::{ElementRef, Html, Selector};
use serde::Deserialize;
use url::Url;

#[derive(Debug, Clone)]
pub(crate) struct GoogleQuery {
    pub text: String,
    pub language: Option<String>,
    pub region: Option<String>,
    pub safe_search: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct GoogleRequestPlan {
    pub url: String,
    pub user_agent: String,
    pub cookie_header: String,
    pub emulation_major: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GoogleTraits {
    pub all_locale: String,
    pub custom: GoogleCustom,
    pub languages: HashMap<String, String>,
    pub regions: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GoogleCustom {
    pub supported_domains: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub(crate) struct GoogleInfo {
    pub subdomain: String,
    pub params: BTreeMap<String, String>,
    pub user_agent: String,
    pub cookies: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedGoogleHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedGoogleResponse {
    pub hits: Vec<ParsedGoogleHit>,
    pub suggestions: Vec<String>,
}

static DATA_IMAGE_RE: OnceLock<Regex> = OnceLock::new();
static SCRIPT_RE: OnceLock<Regex> = OnceLock::new();
static A_RESULT_SELECTOR: OnceLock<Selector> = OnceLock::new();
static TITLE_SELECTOR: OnceLock<Selector> = OnceLock::new();
static IMG_SELECTOR: OnceLock<Selector> = OnceLock::new();
static CONTENT_SELECTOR: OnceLock<Selector> = OnceLock::new();
static MODERN_RESULT_SELECTOR: OnceLock<Selector> = OnceLock::new();
static MODERN_LINK_SELECTOR: OnceLock<Selector> = OnceLock::new();
static MODERN_TITLE_SELECTOR: OnceLock<Selector> = OnceLock::new();
static MODERN_SNIPPET_SELECTOR: OnceLock<Selector> = OnceLock::new();
static SUGGESTION_SELECTOR: OnceLock<Selector> = OnceLock::new();
static TRAITS: OnceLock<std::result::Result<GoogleTraits, String>> = OnceLock::new();

pub(crate) fn build_request_plan(query: &GoogleQuery, page_no: usize) -> Result<GoogleRequestPlan> {
    let sxng_locale = derive_sxng_locale(query);
    let google_info = get_google_info(&sxng_locale, embedded_google_traits()?)?;
    Ok(GoogleRequestPlan {
        url: build_search_url(query, &google_info, page_no),
        user_agent: google_info.user_agent.clone(),
        cookie_header: cookie_header_value(&google_info),
        emulation_major: google_transport_major(&google_info.user_agent),
    })
}

#[cfg(test)]
pub(crate) fn detect_google_interstitial(final_url: &Url, body: &str) -> Result<()> {
    let lowered_url = final_url.as_str().to_ascii_lowercase();
    if lowered_url.contains("sorry.google.com") || final_url.path().starts_with("/sorry") {
        bail!("Google returned a sorry/CAPTCHA page");
    }
    if lowered_url.contains("consent.google.com") {
        bail!("Google returned a consent interstitial");
    }

    let lowered_body = body.to_ascii_lowercase();
    let has_result_markers = lowered_body.contains("data-ved");
    if lowered_body.contains("captcha-form")
        || lowered_body.contains("/sorry/index")
        || lowered_body.contains("consent.google.com")
        || lowered_body.contains("before you continue to google")
        || lowered_body.contains("to continue, please type the characters below")
    {
        bail!("Google returned a consent, sorry, or CAPTCHA interstitial");
    }

    if !has_result_markers
        && (lowered_body.contains("/httpservice/retry/enablejs")
            || lowered_body.contains("enablejs")
            || lowered_body.contains("please click")
                && lowered_body.contains("if you are not redirected within a few seconds"))
    {
        bail!("Google returned a JavaScript-only enablejs interstitial");
    }

    if body.starts_with(")]}'")
        && lowered_body.contains("data-async-context=\"query:\"")
        && !lowered_body.contains("<a ")
    {
        bail!("Google returned an async bootstrap payload without result anchors");
    }

    Ok(())
}

pub(crate) fn parse_response(body: &str) -> ParsedGoogleResponse {
    let data_image_map = parse_url_images(body);
    let document = Html::parse_document(body);
    let mut hits = parse_modern_results(&document);

    let a_result_selector = A_RESULT_SELECTOR
        .get_or_init(|| Selector::parse("a[data-ved]:not([class])").expect("valid selector"));
    let title_selector =
        TITLE_SELECTOR.get_or_init(|| Selector::parse("div[style]").expect("valid selector"));
    let img_selector = IMG_SELECTOR.get_or_init(|| Selector::parse("img").expect("valid selector"));
    let content_selector = CONTENT_SELECTOR
        .get_or_init(|| Selector::parse("div.ilUpNd.H66NU.aSRlid").expect("valid selector"));
    let suggestion_selector = SUGGESTION_SELECTOR
        .get_or_init(|| Selector::parse("div.gGQDvd.iIWm4b a").expect("valid selector"));

    if hits.is_empty() {
        for result in document.select(a_result_selector) {
            let Some(title_node) = result.select(title_selector).next() else {
                continue;
            };
            let title = normalize_ws(&title_node.text().collect::<Vec<_>>().join(" "));
            if title.is_empty() {
                continue;
            }

            let Some(raw_url) = result.value().attr("href") else {
                continue;
            };
            let url = normalize_result_url(raw_url);
            if url.is_empty() || !url.starts_with("http") {
                continue;
            }

            let snippet = extract_snippet(&result, content_selector);

            let _thumbnail = result.select(img_selector).next().and_then(|img| {
                let src = img.value().attr("src")?.to_string();
                if src.starts_with("data:image") {
                    let id = img.value().attr("id")?;
                    data_image_map.get(id).cloned().or(Some(src))
                } else {
                    Some(src)
                }
            });

            hits.push(ParsedGoogleHit {
                title,
                url,
                snippet,
            });
        }
    }

    let suggestions = document
        .select(suggestion_selector)
        .map(|node| normalize_ws(&node.text().collect::<Vec<_>>().join(" ")))
        .filter(|text| !text.is_empty())
        .collect();

    ParsedGoogleResponse { hits, suggestions }
}

fn parse_modern_results(document: &Html) -> Vec<ParsedGoogleHit> {
    let modern_result_selector = MODERN_RESULT_SELECTOR
        .get_or_init(|| Selector::parse("div.N54PNb.BToiNc").expect("valid selector"));
    let modern_link_selector = MODERN_LINK_SELECTOR.get_or_init(|| {
        Selector::parse("a.UBFage[href], div.yuRUbf a[href], a.zReHs[href]")
            .expect("valid selector")
    });
    let modern_title_selector = MODERN_TITLE_SELECTOR
        .get_or_init(|| Selector::parse("h3, div[role='heading']").expect("valid selector"));
    let modern_snippet_selector = MODERN_SNIPPET_SELECTOR.get_or_init(|| {
        Selector::parse("div.VwiC3b, div.VwiC3b.yXK7lf, div.yXK7lf").expect("valid selector")
    });

    let mut hits = Vec::new();
    let mut seen_urls = HashSet::new();
    for result in document.select(modern_result_selector) {
        let Some(link) = result.select(modern_link_selector).next() else {
            continue;
        };
        let Some(raw_url) = link.value().attr("href") else {
            continue;
        };
        let url = normalize_result_url(raw_url);
        if url.is_empty() || !url.starts_with("http") {
            continue;
        }
        if !seen_urls.insert(url.clone()) {
            continue;
        }

        let title = link
            .select(modern_title_selector)
            .next()
            .map(|node| normalize_ws(&node.text().collect::<Vec<_>>().join(" ")))
            .filter(|text| !text.is_empty())
            .unwrap_or_else(|| normalize_ws(&link.text().collect::<Vec<_>>().join(" ")));
        if title.is_empty() {
            continue;
        }

        let snippet = result
            .select(modern_snippet_selector)
            .next()
            .map(extract_node_text)
            .unwrap_or_default();

        hits.push(ParsedGoogleHit {
            title,
            url,
            snippet,
        });
    }

    hits
}

pub(crate) fn apply_window(
    parsed: ParsedGoogleResponse,
    absolute_offset: usize,
    count: usize,
) -> ParsedGoogleResponse {
    let page_skip = absolute_offset % 10;
    ParsedGoogleResponse {
        hits: parsed
            .hits
            .into_iter()
            .skip(page_skip)
            .take(count)
            .collect(),
        suggestions: parsed.suggestions,
    }
}

fn embedded_google_traits() -> Result<&'static GoogleTraits> {
    match TRAITS.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/searxng_google_traits.json"))
            .context("failed to decode embedded SearXNG Google traits")
            .map_err(|err| err.to_string())
    }) {
        Ok(traits) => Ok(traits),
        Err(message) => Err(anyhow::anyhow!("{message}")),
    }
}

fn gen_browser_useragent() -> String {
    if cfg!(target_os = "macos") {
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36".to_string()
    } else if cfg!(target_os = "windows") {
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36".to_string()
    } else {
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36".to_string()
    }
}

fn parse_chrome_major(user_agent: &str) -> Option<u16> {
    let marker = "Chrome/";
    let start = user_agent.find(marker)? + marker.len();
    let version = &user_agent[start..];
    let major = version.split('.').next()?.trim();
    major.parse::<u16>().ok()
}

fn is_supported_transport_major(major: u16) -> bool {
    matches!(
        major,
        100 | 101
            | 104
            | 105
            | 106
            | 107
            | 108
            | 109
            | 110
            | 114
            | 116
            | 117
            | 118
            | 119
            | 120
            | 123
            | 124
            | 126
            | 127
            | 128
            | 129
            | 130
            | 131
            | 132
            | 133
            | 134
            | 135
            | 136
            | 137
            | 138
            | 139
            | 140
            | 141
            | 142
            | 143
            | 144
            | 145
            | 146
    )
}

fn google_transport_major(user_agent: &str) -> u16 {
    let Some(major) = parse_chrome_major(user_agent) else {
        return 146;
    };
    if is_supported_transport_major(major) {
        return major;
    }
    if major > 146 {
        return 146;
    }
    if major >= 136 {
        return 136;
    }
    if major >= 124 {
        return 124;
    }
    if major >= 120 {
        return 120;
    }
    if major >= 114 {
        return 114;
    }
    if major >= 110 {
        return 110;
    }
    if major >= 104 {
        return 104;
    }
    100
}

fn derive_sxng_locale(query: &GoogleQuery) -> String {
    match (query.language.as_deref(), query.region.as_deref()) {
        (Some(lang), Some(region)) => {
            let lang = normalize_locale_tag(lang);
            if lang.contains('-') {
                lang
            } else {
                format!("{}-{}", lang, region.trim().to_ascii_uppercase())
            }
        }
        (Some(lang), None) => normalize_locale_tag(lang),
        (None, Some(region)) => format!("en-{}", region.trim().to_ascii_uppercase()),
        (None, None) => "all".to_string(),
    }
}

fn get_google_info(sxng_locale: &str, traits: &GoogleTraits) -> Result<GoogleInfo> {
    let eng_lang = get_language(traits, sxng_locale, "lang_en");
    let lang_code = eng_lang.split('_').nth(1).unwrap_or("en").to_string();
    let country = get_region(traits, sxng_locale, &traits.all_locale);
    let subdomain = traits
        .custom
        .supported_domains
        .get(&country.to_ascii_uppercase())
        .cloned()
        .unwrap_or_else(|| "www.google.com".to_string());

    let mut params = BTreeMap::new();
    params.insert("hl".to_string(), format!("{}-{}", lang_code, country));
    params.insert(
        "lr".to_string(),
        if sxng_locale == "all" {
            "".to_string()
        } else {
            eng_lang.clone()
        },
    );
    params.insert(
        "cr".to_string(),
        if sxng_locale.contains('-') {
            format!("country{}", country)
        } else {
            "".to_string()
        },
    );
    params.insert("ie".to_string(), "utf8".to_string());
    params.insert("oe".to_string(), "utf8".to_string());

    Ok(GoogleInfo {
        subdomain,
        params,
        user_agent: gen_browser_useragent(),
        cookies: BTreeMap::new(),
    })
}

fn build_search_url(query: &GoogleQuery, google_info: &GoogleInfo, page_no: usize) -> String {
    let start = page_no.saturating_sub(1) * 10;
    let mut url = Url::parse(&format!("https://{}/search", google_info.subdomain))
        .expect("Google subdomain URL must be valid");

    {
        let mut qp = url.query_pairs_mut();
        qp.append_pair("q", &query.text);
        for (key, value) in &google_info.params {
            if !value.trim().is_empty() {
                qp.append_pair(key, value);
            }
        }
        if start > 0 {
            qp.append_pair("start", &start.to_string());
        }

        if let Some(level) = safe_search_value(query.safe_search) {
            qp.append_pair("safe", level);
        }
    }

    url.to_string()
}

fn cookie_header_value(info: &GoogleInfo) -> String {
    info.cookies
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join("; ")
}

fn parse_url_images(text: &str) -> HashMap<String, String> {
    let re = DATA_IMAGE_RE.get_or_init(|| {
        Regex::new(r"(data:image[^']*?)'[^']*?'((?:dimg|pimg|tsuid)[^']*)")
            .expect("valid image regex")
    });

    re.captures_iter(text)
        .filter_map(|caps| {
            let image = caps.get(1)?.as_str();
            let id = caps.get(2)?.as_str();
            Some((id.to_string(), decode_unicode_escapes(image)))
        })
        .collect()
}

fn get_language(traits: &GoogleTraits, sxng_locale: &str, default: &str) -> String {
    if let Some(v) = traits.languages.get(sxng_locale) {
        return v.clone();
    }

    let locale = parse_locale_bits(sxng_locale);
    let language_tag = if let Some(script) = locale.script.as_deref() {
        format!("{}_{}", locale.language, script)
    } else {
        locale.language.clone()
    };

    if let Some(v) = traits.languages.get(&language_tag) {
        return v.clone();
    }
    if let Some(v) = traits.languages.get(&locale.language) {
        return v.clone();
    }

    default.to_string()
}

fn get_region(traits: &GoogleTraits, sxng_locale: &str, default: &str) -> String {
    if let Some(v) = traits.regions.get(sxng_locale) {
        return v.clone();
    }

    let locale = parse_locale_bits(sxng_locale);
    if let Some(territory) = locale.territory.as_deref() {
        let region_key = format!("{}-{}", locale.language, territory);
        if let Some(v) = traits.regions.get(&region_key) {
            return v.clone();
        }
        if traits.custom.supported_domains.contains_key(territory) {
            return territory.to_string();
        }
    }

    let preferred_territory = if locale.language == "en" {
        Some("US".to_string())
    } else {
        Some(locale.language.to_ascii_uppercase())
    };

    if let Some(territory) = preferred_territory {
        let region_key = format!("{}-{}", locale.language, territory);
        if let Some(v) = traits.regions.get(&region_key) {
            return v.clone();
        }
    }

    let mut fallback_regions: Vec<(&String, &String)> = traits
        .regions
        .iter()
        .filter(|(key, _)| key.starts_with(&format!("{}-", locale.language)))
        .collect();
    fallback_regions.sort_by(|a, b| a.0.cmp(b.0));
    if let Some((_, value)) = fallback_regions.first() {
        return (*value).clone();
    }

    default.to_string()
}

fn parse_locale_bits(input: &str) -> LocaleBits {
    if input.trim().is_empty() || input == "all" {
        return LocaleBits {
            language: "en".to_string(),
            script: None,
            territory: None,
        };
    }

    let raw = normalize_locale_tag(input);
    let parts: Vec<&str> = raw.split('-').collect();

    let language = parts
        .first()
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_else(|| "en".to_string());
    let mut script = None;
    let mut territory = None;

    for part in parts.iter().skip(1) {
        if part.len() == 4 {
            let mut chars = part.chars();
            let first = chars.next().unwrap_or('H').to_ascii_uppercase();
            let rest = chars.as_str().to_ascii_lowercase();
            script = Some(format!("{}{}", first, rest));
        } else if part.len() == 2 || part.len() == 3 {
            territory = Some(part.to_ascii_uppercase());
        }
    }

    LocaleBits {
        language,
        script,
        territory,
    }
}

fn normalize_locale_tag(input: &str) -> String {
    let trimmed = input.trim().replace('_', "-");
    let parts: Vec<String> = trimmed
        .split('-')
        .filter(|part| !part.is_empty())
        .enumerate()
        .map(|(idx, part)| {
            if idx == 0 {
                part.to_ascii_lowercase()
            } else if part.len() == 4 {
                let mut chars = part.chars();
                let first = chars.next().unwrap_or('H').to_ascii_uppercase();
                let rest = chars.as_str().to_ascii_lowercase();
                format!("{}{}", first, rest)
            } else {
                part.to_ascii_uppercase()
            }
        })
        .collect();
    parts.join("-")
}

fn safe_search_value(level: u8) -> Option<&'static str> {
    match level {
        0 => None,
        1 => Some("medium"),
        _ => Some("high"),
    }
}

fn normalize_result_url(raw_url: &str) -> String {
    if raw_url.starts_with("/url?q=") {
        let stripped = &raw_url[7..];
        let first = stripped.split("&sa=U").next().unwrap_or(stripped);
        url_decode_lossy(first)
    } else {
        raw_url.to_string()
    }
}

fn url_decode_lossy(input: &str) -> String {
    let replaced = input.replace('+', " ");
    percent_decode(&replaced)
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'%' && idx + 2 < bytes.len() {
            let hex = &input[idx + 1..idx + 3];
            if let Ok(value) = u8::from_str_radix(hex, 16) {
                out.push(value);
                idx += 3;
                continue;
            }
        }
        out.push(bytes[idx]);
        idx += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn decode_unicode_escapes(input: &str) -> String {
    let mut out = String::new();
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }

        match chars.next() {
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                if let Ok(code) = u32::from_str_radix(&hex, 16) {
                    if let Some(decoded) = char::from_u32(code) {
                        out.push(decoded);
                    }
                }
            }
            Some('x') => {
                let hex: String = chars.by_ref().take(2).collect();
                if let Ok(code) = u8::from_str_radix(&hex, 16) {
                    out.push(code as char);
                }
            }
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('\'') => out.push('\''),
            Some(other) => out.push(other),
            None => break,
        }
    }
    out
}

fn extract_snippet(result: &ElementRef<'_>, selector: &Selector) -> String {
    let Some(parent1) = result.parent().and_then(ElementRef::wrap) else {
        return String::new();
    };
    let Some(parent2) = parent1.parent().and_then(ElementRef::wrap) else {
        return String::new();
    };
    let Some(snippet_node) = parent2.select(selector).next() else {
        return String::new();
    };

    let script_re = SCRIPT_RE.get_or_init(|| {
        Regex::new(r"(?is)<script\b[^>]*>.*?</script>").expect("valid script regex")
    });
    let snippet_html = snippet_node.html();
    let cleaned_html = script_re.replace_all(&snippet_html, " ");
    let fragment = Html::parse_fragment(&cleaned_html);
    normalize_ws(&fragment.root_element().text().collect::<Vec<_>>().join(" "))
}

fn extract_node_text(node: ElementRef<'_>) -> String {
    let script_re = SCRIPT_RE.get_or_init(|| {
        Regex::new(r"(?is)<script\b[^>]*>.*?</script>").expect("valid script regex")
    });
    let html = node.html();
    let cleaned_html = script_re.replace_all(&html, " ");
    let fragment = Html::parse_fragment(&cleaned_html);
    normalize_ws(&fragment.root_element().text().collect::<Vec<_>>().join(" "))
}

fn normalize_ws(input: &str) -> String {
    input
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace(" !", "!")
        .replace(" ?", "?")
}

#[derive(Debug, Clone)]
struct LocaleBits {
    language: String,
    script: Option<String>,
    territory: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_google_info_and_url() {
        let query = GoogleQuery {
            text: "rust".to_string(),
            language: Some("de".to_string()),
            region: Some("DE".to_string()),
            safe_search: 2,
        };

        let sxng_locale = derive_sxng_locale(&query);
        let info = get_google_info(&sxng_locale, embedded_google_traits().expect("traits"))
            .expect("google info");
        let url = build_search_url(&query, &info, 2);
        let plan = build_request_plan(&query, 2).expect("request plan");

        assert!(info.user_agent.contains("Chrome/"));
        assert!(url.contains("q=rust"));
        assert!(url.contains("hl=de-DE"));
        assert!(url.contains("lr=lang_de"));
        assert!(url.contains("cr=countryDE"));
        assert!(url.contains("start=10"));
        assert!(url.contains("safe=high"));
        assert!(!url.contains("num="));
        assert!(is_supported_transport_major(plan.emulation_major));
    }

    #[test]
    fn omits_empty_and_default_google_query_params() {
        let query = GoogleQuery {
            text: "rust".to_string(),
            language: None,
            region: None,
            safe_search: 0,
        };

        let info = GoogleInfo {
            subdomain: "www.google.com".to_string(),
            params: BTreeMap::from([
                ("hl".to_string(), "en-US".to_string()),
                ("lr".to_string(), "".to_string()),
                ("cr".to_string(), "".to_string()),
                ("ie".to_string(), "utf8".to_string()),
                ("oe".to_string(), "utf8".to_string()),
            ]),
            user_agent: gen_browser_useragent(),
            cookies: BTreeMap::new(),
        };

        let url = build_search_url(&query, &info, 1);
        assert!(url.contains("q=rust"));
        assert!(url.contains("hl=en-US"));
        assert!(url.contains("ie=utf8"));
        assert!(url.contains("oe=utf8"));
        assert!(!url.contains("lr="));
        assert!(!url.contains("cr="));
        assert!(!url.contains("filter="));
        assert!(!url.contains("start="));
    }

    #[test]
    fn generated_browser_user_agent_looks_like_desktop_chrome() {
        let user_agent = gen_browser_useragent();
        assert!(user_agent.starts_with("Mozilla/5.0"));
        assert!(user_agent.contains(" Chrome/"));
        assert!(user_agent.contains(" Safari/537.36"));
        assert!(parse_chrome_major(&user_agent).is_some());
        assert!(is_supported_transport_major(google_transport_major(
            &user_agent
        )));
    }

    #[test]
    fn parses_fixture_results() {
        let html = include_str!("../fixtures/google_results_fixture.html");
        let parsed = parse_response(html);
        assert_eq!(parsed.hits.len(), 2);
        assert_eq!(parsed.hits[0].title, "Rust Programming Language");
        assert_eq!(parsed.hits[0].url, "https://www.rust-lang.org/");
        assert_eq!(
            parsed.hits[0].snippet,
            "Empowering everyone to build reliable and efficient software."
        );
        assert_eq!(parsed.suggestions, vec!["rust tutorial", "rust async"]);
    }

    #[test]
    fn parses_modern_fixture_results() {
        let html = include_str!("../fixtures/google_results_modern_fixture.html");
        let parsed = parse_response(html);
        assert_eq!(parsed.hits.len(), 2);
        assert_eq!(parsed.hits[0].title, "RFC 9110: HTTP Semantics");
        assert_eq!(
            parsed.hits[0].url,
            "https://www.rfc-editor.org/rfc/rfc9110.html"
        );
        assert_eq!(
            parsed.hits[0].snippet,
            "This document describes the overall architecture of HTTP, establishes common terminology, and defines aspects of the protocol that are shared by all versions."
        );
        assert_eq!(parsed.hits[1].title, "RFC 9110 - HTTP Semantics");
        assert_eq!(
            parsed.hits[1].url,
            "https://datatracker.ietf.org/doc/html/rfc9110"
        );
        assert_eq!(parsed.suggestions, Vec::<String>::new());
    }

    #[test]
    fn parses_mobile_card_fixture_results() {
        let html = include_str!("../fixtures/google_results_mobile_fixture.html");
        let parsed = parse_response(html);
        assert_eq!(parsed.hits.len(), 2);
        assert_eq!(parsed.hits[0].title, "RFC 9110: HTTP Semantics");
        assert_eq!(
            parsed.hits[0].url,
            "https://www.rfc-editor.org/rfc/rfc9110.html"
        );
        assert_eq!(
            parsed.hits[0].snippet,
            "This document describes the overall architecture of HTTP, establishes common terminology, and defines aspects of the protocol that are shared by all versions."
        );
        assert_eq!(parsed.hits[1].title, "create-docusaurus");
        assert_eq!(
            parsed.hits[1].url,
            "https://docusaurus.io/docs/api/misc/create-docusaurus"
        );
        assert_eq!(
            parsed.hits[1].snippet,
            "6 Mar 2026 — create-docusaurus A scaffolding utility to help you instantly set up a functional Docusaurus app. Usage"
        );
    }

    #[test]
    fn detects_google_interstitials() {
        let sorry = Url::parse("https://sorry.google.com/sorry/index").expect("valid url");
        assert!(detect_google_interstitial(&sorry, "<html></html>").is_err());

        let consent =
            Url::parse("https://consent.google.com/ml?continue=https://www.google.com/search")
                .expect("valid url");
        assert!(detect_google_interstitial(&consent, "<html></html>").is_err());

        let enablejs = Url::parse("https://www.google.com/search?q=test").expect("valid url");
        assert!(detect_google_interstitial(
            &enablejs,
            "<html><body><meta content=\"0;url=/httpservice/retry/enablejs?sei=abc\" http-equiv=\"refresh\"></body></html>",
        )
        .is_err());

        let async_stub = Url::parse("https://www.google.com/search?q=test").expect("valid url");
        assert!(detect_google_interstitial(
            &async_stub,
            ")]}'\n33;[\"token\",\"2415\",null,[3,0,1,1,0]]c;[2,null,\"0\"]78c;<div decode-data-ved=\"1\" data-async-context=\"query:\"></div>",
        )
        .is_err());

        let results_with_noscript =
            Url::parse("https://www.google.com/search?q=rust").expect("valid url");
        assert!(detect_google_interstitial(
            &results_with_noscript,
            "<html><body><noscript><meta content=\"0;url=/httpservice/retry/enablejs?sei=abc\" http-equiv=\"refresh\"><div>Please click <a href=\"/httpservice/retry/enablejs?sei=abc\">here</a> if you are not redirected within a few seconds.</div></noscript><a data-ved=\"123\" href=\"https://www.rust-lang.org/\">Rust</a></body></html>",
        )
        .is_ok());
    }
}
