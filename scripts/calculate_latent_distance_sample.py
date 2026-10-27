#!/usr/bin/env python
"""Example: calculate monthly and member-mean latent distance."""

import numpy as np
import pandas as pd

from climatesleuth.evaluation import monthly_latent_distance, summarize_member_distance


def main():
    rng = np.random.default_rng(42)
    feature_columns = [f"z{k}" for k in range(8)]

    # Example model embeddings: 3 members x 12 months.
    model_rows = []
    for member in range(1, 4):
        for month in range(1, 13):
            row = {"member": member, "month": month}
            row.update(dict(zip(feature_columns, rng.normal(size=len(feature_columns)))))
            model_rows.append(row)
    model_embeddings = pd.DataFrame(model_rows)

    # Example reference/observation embeddings: one vector per month.
    ref_rows = []
    for month in range(1, 13):
        row = {"month": month}
        row.update(dict(zip(feature_columns, rng.normal(size=len(feature_columns)))))
        ref_rows.append(row)
    reference_embeddings = pd.DataFrame(ref_rows)

    monthly = monthly_latent_distance(model_embeddings, reference_embeddings, feature_columns=feature_columns)
    summary = summarize_member_distance(monthly)
    print(monthly.head())
    print(summary)


if __name__ == "__main__":
    main()
