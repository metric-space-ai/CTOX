// ref: internal/runtime/executor/xai_executor_media.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::cliproxy::executor::Options;

use super::xai_executor_request::{
    XAI_IMAGES_EDITS_PATH, XAI_IMAGES_GENERATIONS_PATH, XAI_VIDEOS_EDITS_PATH,
    XAI_VIDEOS_EXTENSIONS_PATH, XAI_VIDEOS_GENERATIONS_PATH, XAI_VIDEOS_PATH,
};

#[must_use]
pub fn xai_image_endpoint_path(options: &Options) -> Option<&'static str> {
    let path = options
        .metadata
        .request_path
        .as_deref()
        .unwrap_or_default()
        .trim();
    match path {
        "/v1/images/edits" | "/images/edits" => Some(XAI_IMAGES_EDITS_PATH),
        "/v1/images/generations" | "/images/generations" => Some(XAI_IMAGES_GENERATIONS_PATH),
        _ if options.alt == "images/edits" => Some(XAI_IMAGES_EDITS_PATH),
        _ if options.alt == "images/generations" => Some(XAI_IMAGES_GENERATIONS_PATH),
        _ => None,
    }
}

#[must_use]
pub fn xai_video_endpoint_path(options: &Options) -> Option<String> {
    let path = options
        .metadata
        .request_path
        .as_deref()
        .unwrap_or_default()
        .trim();
    if let Some(native) = path
        .strip_prefix("/v1/videos/")
        .or_else(|| path.strip_prefix("/videos/"))
    {
        return Some(format!("{XAI_VIDEOS_PATH}/{native}"));
    }
    match options.alt.as_str() {
        "videos/generations" => Some(XAI_VIDEOS_GENERATIONS_PATH.into()),
        "videos/edits" => Some(XAI_VIDEOS_EDITS_PATH.into()),
        "videos/extensions" => Some(XAI_VIDEOS_EXTENSIONS_PATH.into()),
        value if value.starts_with("videos/") => Some(format!("/{value}")),
        _ => None,
    }
}

#[must_use]
pub fn xai_is_video_request(options: &Options) -> bool {
    xai_video_endpoint_path(options).is_some()
}
