//! `impressum` — Tier P, DACH first-party legal notices.
//!
//! Unlike portal-backed sources, this source has no fixed website of its own.
//! Its input-driven scrape target discovers the researched company's official
//! domain from the company name (or uses an input URL/domain when available),
//! then extracts evidence from that company's own Impressum.

use super::{Country, FieldKey, ShapedQuery, SourceCtx, SourceModule, Tier};

const ID: &str = "impressum";

struct Impressum;

impl SourceModule for Impressum {
    fn id(&self) -> &'static str {
        ID
    }

    fn scrape_target_key(&self) -> Option<&'static str> {
        Some("impressum")
    }

    fn first_party_evidence(&self) -> bool {
        true
    }

    fn tier(&self) -> Tier {
        Tier::P
    }

    fn countries(&self) -> &'static [Country] {
        &[Country::De, Country::At, Country::Ch]
    }

    fn authoritative_for(&self) -> &'static [FieldKey] {
        &[
            FieldKey::FirmaAnschrift,
            FieldKey::FirmaPlz,
            FieldKey::FirmaOrt,
            FieldKey::FirmaEmail,
            FieldKey::FirmaTelefon,
            FieldKey::FirmaDomain,
            FieldKey::PersonVorname,
            FieldKey::PersonNachname,
            FieldKey::PersonTitel,
            FieldKey::PersonFunktion,
        ]
    }

    fn shape_query(&self, query: &str, _ctx: &SourceCtx<'_>) -> Option<ShapedQuery> {
        let query = query.trim();
        if query.is_empty() {
            return None;
        }
        Some(ShapedQuery {
            query: format!("{query} Impressum"),
            domains: Vec::new(),
        })
    }
}

static MODULE: Impressum = Impressum;

pub fn module() -> &'static dyn SourceModule {
    &MODULE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impressum_declares_scrape_target_and_authoritative_fields() {
        let module = module();
        assert_eq!(module.scrape_target_key(), Some("impressum"));
        assert!(module.first_party_evidence());
        assert_eq!(
            module.authoritative_for(),
            &[
                FieldKey::FirmaAnschrift,
                FieldKey::FirmaPlz,
                FieldKey::FirmaOrt,
                FieldKey::FirmaEmail,
                FieldKey::FirmaTelefon,
                FieldKey::FirmaDomain,
                FieldKey::PersonVorname,
                FieldKey::PersonNachname,
                FieldKey::PersonTitel,
                FieldKey::PersonFunktion,
            ]
        );
    }
}
