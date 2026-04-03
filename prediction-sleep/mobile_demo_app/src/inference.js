export function predictWithLinearContract(inputFeatures, contract) {
  const order = contract.feature_order;
  const med = contract.imputer_median;
  const mean = contract.scaler_mean;
  const scale = contract.scaler_scale;
  const coef = contract.coef;
  const intercept = contract.intercept;
  const classes = contract.classes;
  const threshold = Number.isFinite(Number(contract.decision_threshold))
    ? Number(contract.decision_threshold)
    : 0.5;

  let logit = intercept;
  for (const feature of order) {
    const raw = Number(inputFeatures[feature]);
    const value = Number.isFinite(raw) ? raw : med[feature];
    const denom = scale[feature] === 0 ? 1 : scale[feature];
    const normalized = (value - mean[feature]) / denom;
    logit += normalized * coef[feature];
  }

  const prob1 = 1 / (1 + Math.exp(-logit));
  const prob0 = 1 - prob1;
  const label = prob1 >= threshold ? classes[1] : classes[0];

  return {
    label,
    probabilities: {
      [classes[0]]: prob0,
      [classes[1]]: prob1
    },
    logit
  };
}
