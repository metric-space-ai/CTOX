// ref: internal/config/disable_image_generation_mode_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde::{Deserialize, Serialize};

use super::disable_image_generation_mode::DisableImageGenerationMode;

#[derive(Debug, Deserialize, Serialize)]
struct Wrapper {
    #[serde(rename = "disable-image-generation")]
    value: DisableImageGenerationMode,
}

#[test]
fn unmarshal_yaml_matches_pinned_four_modes() {
    for (source, expected) in [
        ("false", DisableImageGenerationMode::Off),
        ("true", DisableImageGenerationMode::All),
        ("chat", DisableImageGenerationMode::Chat),
        ("passthrough", DisableImageGenerationMode::Passthrough),
    ] {
        let wrapper: Wrapper =
            serde_yaml::from_str(&format!("disable-image-generation: {source}\n")).unwrap();
        assert_eq!(wrapper.value, expected, "source={source}");
        let encoded = serde_yaml::to_string(&wrapper).unwrap();
        let round_trip: Wrapper = serde_yaml::from_str(&encoded).unwrap();
        assert_eq!(round_trip.value, expected);
    }
}

#[test]
fn unmarshal_json_matches_pinned_four_modes() {
    for (source, expected) in [
        ("false", DisableImageGenerationMode::Off),
        ("true", DisableImageGenerationMode::All),
        (r#""chat""#, DisableImageGenerationMode::Chat),
        (r#""passthrough""#, DisableImageGenerationMode::Passthrough),
    ] {
        let mode = DisableImageGenerationMode::parse_json(source.as_bytes()).unwrap();
        assert_eq!(mode, expected, "source={source}");
        assert_eq!(
            serde_json::from_slice::<DisableImageGenerationMode>(source.as_bytes()).unwrap(),
            expected
        );
        assert_eq!(
            serde_json::from_slice::<DisableImageGenerationMode>(
                &serde_json::to_vec(&mode).unwrap()
            )
            .unwrap(),
            expected
        );
    }
}

#[test]
fn aliases_null_empty_and_invalid_values_are_explicit() {
    for alias in ["", "false", "0", "off", "no"] {
        assert_eq!(alias.parse(), Ok(DisableImageGenerationMode::Off));
    }
    for alias in ["true", "1", "on", "yes"] {
        assert_eq!(alias.parse(), Ok(DisableImageGenerationMode::All));
    }
    assert_eq!(
        DisableImageGenerationMode::parse_json(b" \nnull\t"),
        Ok(DisableImageGenerationMode::Off)
    );
    assert_eq!(
        DisableImageGenerationMode::parse_json(b""),
        Ok(DisableImageGenerationMode::Off)
    );
    assert!(DisableImageGenerationMode::parse_json(b"0").is_err());
    assert!("images".parse::<DisableImageGenerationMode>().is_err());
}
