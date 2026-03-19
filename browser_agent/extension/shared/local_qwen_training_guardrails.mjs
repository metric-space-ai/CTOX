function asText(value) {
  return String(value == null ? "" : value).trim();
}

export function getTransformerLoraTrainingSupportIssue(manifest = null) {
  const modules = Array.isArray(manifest?.modules) ? manifest.modules : [];
  const exactTraining =
    manifest?.exactTraining && typeof manifest.exactTraining === "object"
      ? manifest.exactTraining
      : null;
  const browserTraining =
    manifest?.browserTraining && typeof manifest.browserTraining === "object"
      ? manifest.browserTraining
      : null;
  const browserStrategy = asText(browserTraining?.strategy);
  const browserScope = asText(browserTraining?.scope);
  const strategy = asText(exactTraining?.strategy);
  const modulePath = asText(exactTraining?.modulePath);

  if (browserTraining?.enabled === true && browserScope === "all_modules" && browserStrategy) {
    return null;
  }

  if (strategy === "last_down_proj_exact_tail_backward" && modules.length > 1) {
    return {
      code: "single_module_exact_tail_only",
      strategy,
      modulePath,
      moduleCount: modules.length,
      message:
        `Browser transformer LoRA training is disabled: the current ONNX training graph only `
        + `backpropagates through ${modulePath || "a single final module"} while the manifest `
        + `exposes ${modules.length} LoRA modules. This is not normal multi-module LoRA training.`,
    };
  }

  return null;
}
