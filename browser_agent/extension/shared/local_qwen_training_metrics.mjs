function ratePerSecond(count = 0, elapsedMs = 0) {
  const safeCount = Math.max(0, Number(count || 0));
  const safeElapsedMs = Math.max(0, Number(elapsedMs || 0));
  if (!(safeCount > 0) || !(safeElapsedMs > 0)) return 0;
  return safeCount / Math.max(safeElapsedMs / 1000, 1e-6);
}

export function summarizeBrowserTrainingThroughputMetrics({
  totalForwardTokens = 0,
  totalSupervisedTokens = 0,
  trainStepForwardTokens = 0,
  evalForwardTokens = 0,
  trainStepForwardMs = 0,
  evalForwardMs = 0,
  overallElapsedMs = 0,
} = {}) {
  const safeTotalForwardTokens = Math.max(0, Number(totalForwardTokens || 0));
  const safeTotalSupervisedTokens = Math.max(0, Number(totalSupervisedTokens || 0));
  const safeTrainStepForwardTokens = Math.max(0, Number(trainStepForwardTokens || 0));
  const safeEvalForwardTokens = Math.max(0, Number(evalForwardTokens || 0));
  const safeTrainStepForwardMs = Math.max(0, Number(trainStepForwardMs || 0));
  const safeEvalForwardMs = Math.max(0, Number(evalForwardMs || 0));
  const safeOverallElapsedMs = Math.max(0, Number(overallElapsedMs || 0));
  const totalMeasuredForwardMs = safeTrainStepForwardMs + safeEvalForwardMs;
  const nonForwardOverheadMs = Math.max(0, safeOverallElapsedMs - totalMeasuredForwardMs);

  return {
    totalForwardTokens: safeTotalForwardTokens,
    totalSupervisedTokens: safeTotalSupervisedTokens,
    trainStepForwardTokens: safeTrainStepForwardTokens,
    evalForwardTokens: safeEvalForwardTokens,
    trainStepForwardMs: safeTrainStepForwardMs,
    evalForwardMs: safeEvalForwardMs,
    totalMeasuredForwardMs,
    overallElapsedMs: safeOverallElapsedMs,
    nonForwardOverheadMs,
    overallForwardTokensPerSecond: ratePerSecond(safeTotalForwardTokens, safeOverallElapsedMs),
    overallSupervisedTokensPerSecond: ratePerSecond(safeTotalSupervisedTokens, safeOverallElapsedMs),
    measuredForwardTokensPerSecond: ratePerSecond(safeTotalForwardTokens, totalMeasuredForwardMs),
    trainStepForwardTokensPerSecond: ratePerSecond(safeTrainStepForwardTokens, safeTrainStepForwardMs),
    evalForwardTokensPerSecond: ratePerSecond(safeEvalForwardTokens, safeEvalForwardMs),
    forwardWorkloadShare: safeOverallElapsedMs > 0 ? totalMeasuredForwardMs / safeOverallElapsedMs : 0,
    nonForwardOverheadShare: safeOverallElapsedMs > 0 ? nonForwardOverheadMs / safeOverallElapsedMs : 0,
  };
}
