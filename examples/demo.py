"""End-to-end demonstration of smallsurv on a small-sample binary survey dataset."""

import numpy as np
import pandas as pd

from smallsurv import (
    FirthLogisticRegression,
    audit_data,
    bootstrap_optimism_validate,
    evaluate_calibration,
)


def main() -> None:
    rng = np.random.default_rng(2026)
    n = 65

    df = pd.DataFrame(
        {
            "age": rng.normal(38, 11, size=n),
            "income_k": rng.normal(45, 18, size=n),
            "region": rng.choice(["Dhaka", "Chattogram", "Sylhet", "RareZone"], size=n, p=[0.4, 0.3, 0.25, 0.05]),
        }
    )
    logits = -1.2 + 0.04 * (df["age"] - 38) - 0.02 * (df["income_k"] - 45)
    probs_true = 1.0 / (1.0 + np.exp(-logits))
    df["adopted"] = rng.binomial(1, probs_true)

    # 1. Pre-modeling diagnostic audit
    report = audit_data(df, target="adopted", min_category_count=5)
    print(report.summary())

    # 2. Prepare numeric design matrix for Firth/FLIC modeling
    X = pd.get_dummies(df.drop(columns=["adopted"]), drop_first=True, dtype=float)
    y = df["adopted"]

    model = FirthLogisticRegression(flic=True).fit(X, y)
    print("\nFirth / FLIC Coefficient & Odds Ratio Summary:")
    print(model.summary_table().to_string(index=False))

    # 3. Apparent calibration check
    pred_probs = model.predict_proba(X)[:, 1]
    cal = evaluate_calibration(y, pred_probs, n_bins=5)
    print("\n" + cal.summary())

    # 4. Bootstrap optimism-corrected internal validation
    val = bootstrap_optimism_validate(model, X, y, n_bootstraps=50, random_state=2026)
    print("\n" + val.summary())


if __name__ == "__main__":
    main()
