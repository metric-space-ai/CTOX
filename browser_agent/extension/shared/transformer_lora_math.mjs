function createLcg(seed) {
  let state = (seed >>> 0) || 1;
  return function next() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function createNormalSampler(seed) {
  const nextUniform = createLcg(seed);
  let spare = null;
  return function nextNormal() {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u <= 1e-7) u = nextUniform();
    while (v <= 1e-7) v = nextUniform();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    const angle = 2.0 * Math.PI * v;
    spare = mag * Math.sin(angle);
    return mag * Math.cos(angle);
  };
}

function adamUpdate(weights, grads, moments1, moments2, step, learningRate) {
  const beta1 = 0.9;
  const beta2 = 0.999;
  const epsilon = 1e-8;
  const correction1 = 1 - beta1 ** step;
  const correction2 = 1 - beta2 ** step;
  for (let index = 0; index < weights.length; index += 1) {
    const grad = grads[index];
    const m = beta1 * moments1[index] + (1 - beta1) * grad;
    const v = beta2 * moments2[index] + (1 - beta2) * grad * grad;
    moments1[index] = m;
    moments2[index] = v;
    const mHat = m / correction1;
    const vHat = v / correction2;
    weights[index] -= learningRate * (mHat / (Math.sqrt(vHat) + epsilon));
  }
}

function makeModuleSeed(seed, offset) {
  return ((seed >>> 0) + ((offset + 1) * 2654435761 >>> 0)) >>> 0;
}

function moduleStateKey(modulePath = "") {
  return String(modulePath || "").trim();
}

function nextRademacher(nextUniform) {
  return nextUniform() < 0.5 ? -1 : 1;
}

const transformerLoraMutationVersions = new WeakMap();

function initializeTransformerLoraMutationVersion(adapter) {
  if (adapter && typeof adapter === "object") {
    transformerLoraMutationVersions.set(adapter, 0);
  }
  return adapter;
}

function bumpTransformerLoraMutationVersion(adapter) {
  if (!adapter || typeof adapter !== "object") return;
  transformerLoraMutationVersions.set(
    adapter,
    (transformerLoraMutationVersions.get(adapter) || 0) + 1,
  );
}

export function getTransformerLoraMutationVersion(adapter) {
  return transformerLoraMutationVersions.get(adapter) || 0;
}

export function createTransformerLoraAdapter({
  manifest,
  seed = 42,
  initStd = 0.02,
  trainA = true,
  trainB = true,
  modulePaths = null,
} = {}) {
  const manifestModules = Array.isArray(manifest?.modules) ? manifest.modules : [];
  const allowedPaths = new Set(
    Array.isArray(modulePaths) ? modulePaths.map((value) => String(value || "").trim()).filter(Boolean) : []
  );
  const modules = [];

  manifestModules.forEach((entry, index) => {
    const modulePath = moduleStateKey(entry?.modulePath);
    if (!modulePath) return;
    if (allowedPaths.size > 0 && !allowedPaths.has(modulePath)) return;
    const inFeatures = Number(entry?.inFeatures || 0);
    const outFeatures = Number(entry?.outFeatures || 0);
    const rank = Number(entry?.rank || manifest?.graph?.rank || 0);
    if (inFeatures <= 0 || outFeatures <= 0 || rank <= 0) return;
    const nextNormal = createNormalSampler(makeModuleSeed(seed, index));
    const loraA = new Float32Array(inFeatures * rank);
    const loraB = new Float32Array(rank * outFeatures);
    for (let i = 0; i < loraA.length; i += 1) {
      loraA[i] = nextNormal() * initStd;
    }
    modules.push({
      modulePath,
      targetModule: String(entry?.targetModule || "").trim(),
      inFeatures,
      outFeatures,
      rank,
      trainA: trainA !== false,
      trainB: trainB !== false,
      loraInputA: String(entry?.loraInputA || "").trim(),
      loraInputB: String(entry?.loraInputB || "").trim(),
      trainingProbeInput: String(entry?.trainingProbeInput || entry?.activationInputName || "").trim(),
      trainingProbeBaseOutput: String(entry?.trainingProbeBaseOutput || entry?.baseOutputName || "").trim(),
      trainingProbeMergedOutput: String(entry?.trainingProbeMergedOutput || entry?.mergedOutputName || "").trim(),
      loraA,
      loraB,
      scale: 1,
    });
  });

  return initializeTransformerLoraMutationVersion({
    kind: "transformer_lora",
    rank: Number(manifest?.graph?.rank || 0),
    runtimeModelId: String(manifest?.runtimeModelId || "").trim(),
    modules,
  });
}

export function cloneTransformerLoraAdapter(adapter) {
  return initializeTransformerLoraMutationVersion({
    kind: "transformer_lora",
    rank: Number(adapter?.rank || 0),
    runtimeModelId: String(adapter?.runtimeModelId || "").trim(),
    modules: (Array.isArray(adapter?.modules) ? adapter.modules : []).map((module) => ({
      ...module,
      loraA: new Float32Array(module?.loraA || []),
      loraB: new Float32Array(module?.loraB || []),
    })),
  });
}

export function createTransformerLoraAdamState(adapter) {
  const modules = {};
  for (const module of Array.isArray(adapter?.modules) ? adapter.modules : []) {
    const key = moduleStateKey(module?.modulePath);
    if (!key) continue;
    modules[key] = {
      mA: module.trainA === false ? new Float32Array(0) : new Float32Array(module.loraA.length),
      vA: module.trainA === false ? new Float32Array(0) : new Float32Array(module.loraA.length),
      mB: module.trainB === false ? new Float32Array(0) : new Float32Array(module.loraB.length),
      vB: module.trainB === false ? new Float32Array(0) : new Float32Array(module.loraB.length),
      step: 0,
    };
  }
  return {
    kind: "transformer_lora",
    modules,
  };
}

export function createTransformerLoraGradientBuffers(module) {
  return {
    gradA: module?.trainA === false ? new Float32Array(0) : new Float32Array(module.inFeatures * module.rank),
    gradB: module?.trainB === false ? new Float32Array(0) : new Float32Array(module.rank * module.outFeatures),
    lowRank: new Float32Array(0),
    gradLowRank: new Float32Array(0),
  };
}

export function computeTransformerLoraLowRank(inputs, rowCount, module, target = null) {
  const safeRowCount = Math.max(0, Number(rowCount || 0));
  const out = target ?? new Float32Array(safeRowCount * module.rank);
  for (let row = 0; row < safeRowCount; row += 1) {
    const inputOffset = row * module.inFeatures;
    const lowRankOffset = row * module.rank;
    for (let rankIndex = 0; rankIndex < module.rank; rankIndex += 1) {
      let sum = 0.0;
      for (let feature = 0; feature < module.inFeatures; feature += 1) {
        sum += inputs[inputOffset + feature] * module.loraA[feature * module.rank + rankIndex];
      }
      out[lowRankOffset + rankIndex] = sum;
    }
  }
  return out;
}

export function computeTransformerLoraDelta(inputs, rowCount, module, scale = 1, target = null) {
  const safeRowCount = Math.max(0, Number(rowCount || 0));
  const out = target ?? new Float32Array(safeRowCount * module.outFeatures);
  const lowRank = computeTransformerLoraLowRank(inputs, safeRowCount, module);
  const scaled = Math.fround(Number(scale || module?.scale || 1));
  for (let row = 0; row < safeRowCount; row += 1) {
    const lowRankOffset = row * module.rank;
    const outOffset = row * module.outFeatures;
    for (let outFeature = 0; outFeature < module.outFeatures; outFeature += 1) {
      let sum = 0.0;
      for (let rankIndex = 0; rankIndex < module.rank; rankIndex += 1) {
        sum += lowRank[lowRankOffset + rankIndex] * module.loraB[rankIndex * module.outFeatures + outFeature];
      }
      out[outOffset + outFeature] = Math.fround(sum * scaled);
    }
  }
  return out;
}

export function accumulateTransformerLoraModuleGradients({
  inputs,
  gradOutputs,
  rowCount,
  module,
  scale = 1,
  gradA = null,
  gradB = null,
  lowRank = null,
  gradLowRank = null,
  normalizeByRowCount = true,
} = {}) {
  const safeRowCount = Math.max(0, Number(rowCount || 0));
  const safeGradA = gradA ?? (module.trainA === false ? new Float32Array(0) : new Float32Array(module.inFeatures * module.rank));
  const safeGradB = gradB ?? (module.trainB === false ? new Float32Array(0) : new Float32Array(module.rank * module.outFeatures));
  const safeLowRank = computeTransformerLoraLowRank(
    inputs,
    safeRowCount,
    module,
    lowRank ?? new Float32Array(safeRowCount * module.rank),
  );
  const safeGradLowRank = gradLowRank ?? new Float32Array(safeRowCount * module.rank);
  if (safeGradA.length) safeGradA.fill(0);
  if (safeGradB.length) safeGradB.fill(0);
  safeGradLowRank.fill(0);

  const denom = normalizeByRowCount && safeRowCount > 0 ? safeRowCount : 1;
  const scaled = Math.fround(Number(scale || module?.scale || 1) / denom);

  for (let row = 0; row < safeRowCount; row += 1) {
    const inputOffset = row * module.inFeatures;
    const gradOutputOffset = row * module.outFeatures;
    const lowRankOffset = row * module.rank;

    for (let outFeature = 0; outFeature < module.outFeatures; outFeature += 1) {
      const grad = Math.fround(gradOutputs[gradOutputOffset + outFeature] * scaled);
      for (let rankIndex = 0; rankIndex < module.rank; rankIndex += 1) {
        const bIndex = rankIndex * module.outFeatures + outFeature;
        if (safeGradB.length) {
          safeGradB[bIndex] += safeLowRank[lowRankOffset + rankIndex] * grad;
        }
        if (safeGradA.length) {
          safeGradLowRank[lowRankOffset + rankIndex] += grad * module.loraB[bIndex];
        }
      }
    }

    if (safeGradA.length) {
      for (let feature = 0; feature < module.inFeatures; feature += 1) {
        const aOffset = feature * module.rank;
        const inputValue = inputs[inputOffset + feature];
        for (let rankIndex = 0; rankIndex < module.rank; rankIndex += 1) {
          safeGradA[aOffset + rankIndex] += inputValue * safeGradLowRank[lowRankOffset + rankIndex];
        }
      }
    }
  }

  return {
    gradA: safeGradA,
    gradB: safeGradB,
    lowRank: safeLowRank,
    gradLowRank: safeGradLowRank,
  };
}

export function applyTransformerLoraModuleAdamStep({
  module,
  adamState,
  gradA,
  gradB,
  learningRate,
} = {}) {
  if (!module || !adamState) {
    throw new Error("Transformer LoRA Adam step requires both module and adamState.");
  }
  adamState.step = Number(adamState.step || 0) + 1;
  if (module.trainA !== false && gradA?.length) {
    adamUpdate(module.loraA, gradA, adamState.mA, adamState.vA, adamState.step, learningRate);
  }
  if (module.trainB !== false && gradB?.length) {
    adamUpdate(module.loraB, gradB, adamState.mB, adamState.vB, adamState.step, learningRate);
  }
  return adamState;
}

export function perturbTransformerLoraAdapter({
  adapter,
  seed = 0,
  epsilon = 0,
  multiplier = 1,
} = {}) {
  const safeDelta = Math.fround(Number(epsilon || 0) * Number(multiplier || 0));
  if (!safeDelta || !Array.isArray(adapter?.modules)) return adapter;

  for (let moduleIndex = 0; moduleIndex < adapter.modules.length; moduleIndex += 1) {
    const module = adapter.modules[moduleIndex];
    const nextUniform = createLcg(makeModuleSeed(Number(seed) >>> 0, moduleIndex));
    if (module?.trainA !== false && module?.loraA instanceof Float32Array) {
      for (let index = 0; index < module.loraA.length; index += 1) {
        module.loraA[index] = Math.fround(module.loraA[index] + (safeDelta * nextRademacher(nextUniform)));
      }
    }
    if (module?.trainB !== false && module?.loraB instanceof Float32Array) {
      for (let index = 0; index < module.loraB.length; index += 1) {
        module.loraB[index] = Math.fround(module.loraB[index] + (safeDelta * nextRademacher(nextUniform)));
      }
    }
  }

  bumpTransformerLoraMutationVersion(adapter);
  return adapter;
}

export function applyTransformerLoraSpsaAdamStep({
  adapter,
  adam,
  seed = 0,
  epsilon = 0,
  objectiveDiff = 0,
  learningRate = 0,
  gradientClip = 0,
} = {}) {
  const safeEpsilon = Number(epsilon || 0);
  if (!(safeEpsilon > 0)) {
    throw new Error("Transformer LoRA SPSA updates require a positive epsilon.");
  }
  const rawScale = Number(objectiveDiff || 0) / (2 * safeEpsilon);
  const maxAbsScale = Number(gradientClip || 0);
  const gradScale =
    maxAbsScale > 0
      ? Math.max(-maxAbsScale, Math.min(maxAbsScale, rawScale))
      : rawScale;

  for (let moduleIndex = 0; moduleIndex < (Array.isArray(adapter?.modules) ? adapter.modules.length : 0); moduleIndex += 1) {
    const module = adapter.modules[moduleIndex];
    const state = adam?.modules?.[moduleStateKey(module?.modulePath)];
    if (!module || !state) continue;
    const nextUniform = createLcg(makeModuleSeed(Number(seed) >>> 0, moduleIndex));
    const gradA =
      module.trainA === false
        ? new Float32Array(0)
        : new Float32Array(module.loraA.length);
    const gradB =
      module.trainB === false
        ? new Float32Array(0)
        : new Float32Array(module.loraB.length);

    for (let index = 0; index < gradA.length; index += 1) {
      gradA[index] = Math.fround(gradScale * nextRademacher(nextUniform));
    }
    for (let index = 0; index < gradB.length; index += 1) {
      gradB[index] = Math.fround(gradScale * nextRademacher(nextUniform));
    }

    applyTransformerLoraModuleAdamStep({
      module,
      adamState: state,
      gradA,
      gradB,
      learningRate,
    });
  }
  bumpTransformerLoraMutationVersion(adapter);
}

export function trainTransformerLoraModuleFromGradients({
  adapter,
  adam,
  modulePath,
  inputs,
  gradOutputs,
  rowCount,
  learningRate,
  scale = 1,
  normalizeByRowCount = true,
} = {}) {
  const module = (Array.isArray(adapter?.modules) ? adapter.modules : []).find(
    (entry) => moduleStateKey(entry?.modulePath) === moduleStateKey(modulePath),
  );
  if (!module) {
    throw new Error(`Unknown transformer LoRA module: ${String(modulePath || "").trim()}`);
  }
  const state = adam?.modules?.[moduleStateKey(module.modulePath)];
  if (!state) {
    throw new Error(`Missing Adam state for transformer LoRA module: ${module.modulePath}`);
  }
  const gradients = accumulateTransformerLoraModuleGradients({
    inputs,
    gradOutputs,
    rowCount,
    module,
    scale,
    normalizeByRowCount,
  });
  applyTransformerLoraModuleAdamStep({
    module,
    adamState: state,
    gradA: gradients.gradA,
    gradB: gradients.gradB,
    learningRate,
  });
  bumpTransformerLoraMutationVersion(adapter);
  return gradients;
}
