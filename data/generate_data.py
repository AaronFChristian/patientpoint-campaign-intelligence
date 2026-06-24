"""
data/generate_data.py

Generates a synthetic PatientPoint-style campaign analytics dataset.
Outputs 5 CSV files to the data/ directory (anomaly_log starts empty — Agent 2 populates it).

Run from project root:
    python data/generate_data.py

Expected output:
    data/raw_campaigns.csv            ~50 rows
    data/raw_physician_offices.csv    ~5,000 rows
    data/raw_campaign_placements.csv  ~12,500 rows
    data/raw_daily_metrics.csv        ~500,000 rows
    data/raw_weekly_benchmarks.csv    ~2,500 rows
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import date, timedelta

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ── Output directory (same folder as this file) ────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Domain constants ───────────────────────────────────────────────────────────
HEALTH_CONDITIONS = [
    "diabetes", "cardiovascular", "oncology", "respiratory",
    "neurology", "dermatology", "orthopedics", "mental_health",
]
SPECIALTIES = [
    "cardiology", "oncology", "primary_care", "neurology",
    "dermatology", "orthopedics", "psychiatry", "pulmonology", "family_medicine",
]
SCREEN_TYPES = ["waiting_room", "exam_room"]

REGIONS = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NY", "NJ", "PA"],
    "Midwest":   ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    "South":     ["AL", "AR", "DE", "FL", "GA", "KY", "LA", "MD", "MS",
                  "NC", "OK", "SC", "TN", "TX", "VA", "WV"],
    "West":      ["AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"],
}
STATE_TO_REGION = {state: region for region, states in REGIONS.items() for state in states}
ALL_STATES = [s for states in REGIONS.values() for s in states]

# Beta distribution params (a, b) per office tier
# Derived from: a = mean * k, b = (1-mean) * k, where k = mean*(1-mean)/variance - 1
# Tier A → mean ≈ 0.72, Tier B → mean ≈ 0.62, Tier C → mean ≈ 0.52
TIER_BETA = {
    "A": (13.8, 5.36),
    "B": (5.87, 3.60),
    "C": (3.48, 3.22),
}

# Impressions per day range by tier (PatientPoint offices vary by patient volume)
TIER_IMPRESSIONS = {
    "A": (180, 320),
    "B": (90, 180),
    "C": (30, 90),
}

# Tier distribution: most offices are B/C, fewer are top-tier A
TIER_WEIGHTS = {"A": 0.15, "B": 0.55, "C": 0.30}

# Date range: full calendar year (gives 52 weeks for 90-day rolling baselines)
DATE_START = date(2025, 1, 1)
DATE_END   = date(2025, 12, 31)


# ── 1. Campaigns ──────────────────────────────────────────────────────────────
def generate_campaigns(n: int = 50) -> pd.DataFrame:
    """
    Generate campaign records. 48 regular campaigns + 2 named A/B demo campaigns.
    The A/B pair (CAMP_AB_HIGH, CAMP_AB_LOW) have different completion rate distributions
    so Tab 4 always produces a statistically significant result during demos.
    """
    records = []

    # 48 regular campaigns
    for i in range(1, n - 1):
        duration_weeks = random.randint(4, 16)
        max_start_offset = (DATE_END - DATE_START).days - duration_weeks * 7
        start_offset = random.randint(0, max_start_offset)
        start = DATE_START + timedelta(days=start_offset)
        end   = start + timedelta(weeks=duration_weeks)

        records.append({
            "campaign_id":       f"CAMP_{i:03d}",
            "name":              f"{fake.company()} {random.choice(HEALTH_CONDITIONS).title()} Campaign",
            "health_condition":  random.choice(HEALTH_CONDITIONS),
            "specialty_target":  random.choice(SPECIALTIES),
            "start_date":        start,
            "end_date":          end,
            "budget":            round(random.uniform(50_000, 500_000), 2),
            "target_completion": round(random.uniform(0.60, 0.78), 3),
            "is_ab_demo":        False,
        })

    # 2 seeded A/B demo campaigns — same specialty, same duration, different performance
    ab_start = DATE_START + timedelta(weeks=4)
    ab_end   = ab_start  + timedelta(weeks=12)
    records.append({
        "campaign_id":       "CAMP_AB_HIGH",
        "name":              "CardioPlus Advanced — Cardiovascular Campaign",
        "health_condition":  "cardiovascular",
        "specialty_target":  "cardiology",
        "start_date":        ab_start,
        "end_date":          ab_end,
        "budget":            250_000.00,
        "target_completion": 0.70,
        "is_ab_demo":        True,
    })
    records.append({
        "campaign_id":       "CAMP_AB_LOW",
        "name":              "CardioBasic Standard — Cardiovascular Campaign",
        "health_condition":  "cardiovascular",
        "specialty_target":  "cardiology",
        "start_date":        ab_start,
        "end_date":          ab_end,
        "budget":            180_000.00,
        "target_completion": 0.64,
        "is_ab_demo":        True,
    })

    return pd.DataFrame(records)


# ── 2. Physician Offices ───────────────────────────────────────────────────────
def generate_physician_offices(n: int = 5000) -> pd.DataFrame:
    """
    Generate physician office records distributed across US states/regions.
    Tier assignment follows PatientPoint's real network: most offices are mid-tier (B)
    with a long tail of smaller practices (C) and a minority of high-volume offices (A).
    """
    tiers  = random.choices(list(TIER_WEIGHTS.keys()), weights=list(TIER_WEIGHTS.values()), k=n)
    states = random.choices(ALL_STATES, k=n)

    records = [{
        "office_id":  f"OFF_{i:05d}",
        "name":       f"Dr. {fake.last_name()} {random.choice(SPECIALTIES).replace('_', ' ').title()} Associates",
        "specialty":  random.choice(SPECIALTIES),
        "state":      states[i],
        "region":     STATE_TO_REGION[states[i]],
        "tier":       tiers[i],
        "city":       fake.city(),
    } for i in range(n)]

    return pd.DataFrame(records)


# ── 3. Campaign Placements ────────────────────────────────────────────────────
def generate_campaign_placements(
    campaigns: pd.DataFrame,
    offices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Each campaign is placed in a subset of offices. Larger-budget campaigns
    reach more offices. Each placement gets one screen type.
    Includes denormalized campaign dates so metrics generation avoids extra joins.
    """
    records = []
    placement_id = 1
    office_ids = offices["office_id"].values

    for _, c in campaigns.iterrows():
        # Scale office count with budget: ~$200 per office per week as a rough proxy
        weeks_active = max(1, (c["end_date"] - c["start_date"]).days // 7)
        n_offices = int(np.clip(c["budget"] / (200 * weeks_active), 100, 600))
        selected_offices = np.random.choice(office_ids, size=n_offices, replace=False)

        for off_id in selected_offices:
            tier = offices.loc[offices["office_id"] == off_id, "tier"].values[0]
            records.append({
                "placement_id":    f"PL_{placement_id:06d}",
                "campaign_id":     c["campaign_id"],
                "office_id":       off_id,
                "screen_type":     random.choice(SCREEN_TYPES),
                "tier":            tier,
                "campaign_start":  c["start_date"],
                "campaign_end":    c["end_date"],
                "is_ab_high":      c["campaign_id"] == "CAMP_AB_HIGH",
                "is_ab_low":       c["campaign_id"] == "CAMP_AB_LOW",
            })
            placement_id += 1

    return pd.DataFrame(records)


# ── 4. Daily Metrics ──────────────────────────────────────────────────────────
def generate_daily_metrics(placements: pd.DataFrame) -> pd.DataFrame:
    """
    Core dataset: one row per (placement, active day). ~500K rows total.

    Completion rates drawn from a beta distribution per office tier so the
    distribution is realistic (bell-shaped, bounded 0–1) rather than uniform.

    A/B demo campaigns are assigned tighter, deliberately separated distributions
    so the Tab 4 t-test always yields p < 0.05.
    """
    all_ids, all_dates, all_tiers, all_screen, all_ab_high, all_ab_low = [], [], [], [], [], []

    print("  Expanding placements to daily rows...")
    for _, p in placements.iterrows():
        date_range = pd.date_range(p["campaign_start"], p["campaign_end"], freq="D")
        n = len(date_range)
        all_ids.extend([p["placement_id"]] * n)
        all_dates.extend(date_range.tolist())
        all_tiers.extend([p["tier"]] * n)
        all_screen.extend([p["screen_type"]] * n)
        all_ab_high.extend([p["is_ab_high"]] * n)
        all_ab_low.extend([p["is_ab_low"]] * n)

    n_total = len(all_ids)
    print(f"  Generating {n_total:,} metric rows...")

    # ── Impressions: sampled per tier ──────────────────────────────────────────
    tiers_arr = np.array(all_tiers)
    imp_low  = np.array([TIER_IMPRESSIONS[t][0] for t in all_tiers])
    imp_high = np.array([TIER_IMPRESSIONS[t][1] for t in all_tiers])
    impressions = np.random.randint(imp_low, imp_high + 1)

    # ── Completion rates: beta distribution per tier ───────────────────────────
    completion_rates = np.zeros(n_total)
    for tier, (a, b) in TIER_BETA.items():
        mask = tiers_arr == tier
        if mask.any():
            completion_rates[mask] = np.random.beta(a, b, size=mask.sum())

    # Override A/B demo campaigns with tight, separated distributions
    ab_high_mask = np.array(all_ab_high)
    ab_low_mask  = np.array(all_ab_low)
    # CAMP_AB_HIGH: mean=0.71, tight std (a=22.1, b=9.0)
    completion_rates[ab_high_mask] = np.random.beta(22.1, 9.0, size=ab_high_mask.sum())
    # CAMP_AB_LOW: mean=0.63, tight std (a=22.3, b=13.1)
    completion_rates[ab_low_mask]  = np.random.beta(22.3, 13.1, size=ab_low_mask.sum())

    # Clip to valid range
    completion_rates = np.clip(completion_rates, 0.05, 0.99)

    # ── Compute completions, skips, CTR ───────────────────────────────────────
    completions = (impressions * completion_rates).astype(int)
    skips = np.maximum(0, impressions - completions - np.random.randint(0, 10, size=n_total))

    # CTR: waiting room screens get slightly higher CTR (patients have time to interact)
    screen_arr = np.array(all_screen)
    base_ctr   = np.where(screen_arr == "waiting_room",
                          np.random.uniform(0.03, 0.09, n_total),
                          np.random.uniform(0.01, 0.05, n_total))
    ctr = np.round(base_ctr, 4)

    metric_ids = [f"MET_{i+1:07d}" for i in range(n_total)]

    return pd.DataFrame({
        "metric_id":       metric_ids,
        "placement_id":    all_ids,
        "date":            all_dates,
        "impressions":     impressions,
        "completions":     completions,
        "skips":           skips,
        "ctr":             ctr,
        "completion_rate": np.round(completion_rates, 4),
    })


# ── 5. Seed anomalies into daily_metrics ──────────────────────────────────────
def seed_anomalies(metrics: pd.DataFrame, placements: pd.DataFrame) -> pd.DataFrame:
    """
    For each campaign, select 3–4 placements and insert a 7–10 day window where
    completion_rate drops sharply. These are the signals Agent 2 should detect.

    Two anomaly types:
      'drop'  → completion_rate multiplied by 0.40–0.55 (genuine underperformance)
      'gap'   → impressions set to 0 (data ingestion gap)
    """
    metrics = metrics.copy()
    metrics["anomaly_type"] = None  # tag for our own reference (not used by agents)

    campaign_ids = placements["campaign_id"].unique()

    for camp_id in campaign_ids:
        camp_placements = placements[placements["campaign_id"] == camp_id]["placement_id"].values
        if len(camp_placements) < 4:
            continue

        n_anomalies = random.randint(3, 4)
        selected = np.random.choice(camp_placements, size=n_anomalies, replace=False)

        for pl_id in selected:
            pl_metrics = metrics[metrics["placement_id"] == pl_id]
            if len(pl_metrics) < 14:
                continue

            # Pick a random 7–10 day window within the placement's active period
            window_start_idx = random.randint(7, len(pl_metrics) - 10)
            window_len = random.randint(7, 10)
            window_mask = (
                (metrics["placement_id"] == pl_id) &
                (metrics.index.isin(pl_metrics.index[window_start_idx : window_start_idx + window_len]))
            )

            anomaly_type = random.choice(["drop", "drop", "gap"])  # 2:1 ratio, more drops

            if anomaly_type == "drop":
                drop_multiplier = random.uniform(0.38, 0.52)
                metrics.loc[window_mask, "completion_rate"] = (
                    metrics.loc[window_mask, "completion_rate"] * drop_multiplier
                ).round(4)
                metrics.loc[window_mask, "completions"] = (
                    metrics.loc[window_mask, "impressions"] *
                    metrics.loc[window_mask, "completion_rate"]
                ).astype(int)
            else:  # gap
                metrics.loc[window_mask, "impressions"]     = 0
                metrics.loc[window_mask, "completions"]     = 0
                metrics.loc[window_mask, "skips"]           = 0
                metrics.loc[window_mask, "ctr"]             = 0.0
                metrics.loc[window_mask, "completion_rate"] = 0.0

            metrics.loc[window_mask, "anomaly_type"] = anomaly_type

    n_anomalous = (metrics["anomaly_type"].notna()).sum()
    print(f"  Seeded anomalies in {n_anomalous:,} metric rows across all campaigns")
    return metrics


# ── 6. Weekly Benchmarks ──────────────────────────────────────────────────────
def generate_weekly_benchmarks(
    campaigns: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (campaign, week). Computes actual weekly avg completion rate from
    metrics data, then sets target slightly above national avg — mirrors how a
    real analytics team tracks performance against client KPIs.
    """
    # Attach campaign_id to metrics via placements lookup (use metric placement prefix)
    # Simpler: aggregate by week using campaign dates from the campaigns table
    records = []
    national_avg_base = 0.622  # baseline — varies slightly by week

    for _, c in campaigns.iterrows():
        weeks = pd.date_range(c["start_date"], c["end_date"], freq="W-MON")
        for week_num, week_start in enumerate(weeks, 1):
            week_end = week_start + timedelta(days=6)
            # National avg drifts slightly over the year (seasonal patterns)
            seasonal_drift = 0.01 * np.sin(2 * np.pi * week_num / 52)
            national_avg = round(national_avg_base + seasonal_drift + random.uniform(-0.005, 0.005), 4)
            target = round(c["target_completion"] + random.uniform(-0.01, 0.01), 4)

            records.append({
                "benchmark_id":          f"BM_{c['campaign_id']}_{week_num:03d}",
                "campaign_id":           c["campaign_id"],
                "week_start":            week_start.date(),
                "week_end":              week_end.date(),
                "week_number":           week_num,
                "target_completion_rate": target,
                "national_avg":          national_avg,
            })

    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PatientPoint Campaign Intelligence — Data Generation")
    print("=" * 60)

    print("\n[1/5] Generating campaigns...")
    campaigns = generate_campaigns(n=50)
    print(f"  Created {len(campaigns):,} campaigns ({(campaigns['is_ab_demo']).sum()} A/B demo pair)")

    print("\n[2/5] Generating physician offices...")
    offices = generate_physician_offices(n=5000)
    tier_counts = offices["tier"].value_counts().to_dict()
    print(f"  Created {len(offices):,} offices | Tier A: {tier_counts.get('A',0)}, B: {tier_counts.get('B',0)}, C: {tier_counts.get('C',0)}")

    print("\n[3/5] Generating campaign placements...")
    placements = generate_campaign_placements(campaigns, offices)
    print(f"  Created {len(placements):,} placements across {placements['campaign_id'].nunique()} campaigns")

    print("\n[4/5] Generating daily metrics...")
    metrics = generate_daily_metrics(placements)
    metrics = seed_anomalies(metrics, placements)
    print(f"  Created {len(metrics):,} daily metric rows")

    print("\n[5/5] Generating weekly benchmarks...")
    benchmarks = generate_weekly_benchmarks(campaigns, metrics)
    print(f"  Created {len(benchmarks):,} weekly benchmark rows")

    # ── Save to CSV ────────────────────────────────────────────────────────────
    print("\n[Saving CSVs to data/]")
    # Drop internal helper columns before saving
    placements_clean = placements.drop(columns=["campaign_start", "campaign_end", "is_ab_high", "is_ab_low"])
    metrics_clean    = metrics.drop(columns=["anomaly_type"])

    campaigns.to_csv(os.path.join(DATA_DIR, "raw_campaigns.csv"), index=False)
    offices.to_csv(os.path.join(DATA_DIR, "raw_physician_offices.csv"), index=False)
    placements_clean.to_csv(os.path.join(DATA_DIR, "raw_campaign_placements.csv"), index=False)
    metrics_clean.to_csv(os.path.join(DATA_DIR, "raw_daily_metrics.csv"), index=False)
    benchmarks.to_csv(os.path.join(DATA_DIR, "raw_weekly_benchmarks.csv"), index=False)

    print("\n" + "=" * 60)
    print("DONE — Validation summary")
    print("=" * 60)
    print(f"  raw_campaigns.csv            {len(campaigns):>8,} rows")
    print(f"  raw_physician_offices.csv    {len(offices):>8,} rows")
    print(f"  raw_campaign_placements.csv  {len(placements_clean):>8,} rows")
    print(f"  raw_daily_metrics.csv        {len(metrics_clean):>8,} rows")
    print(f"  raw_weekly_benchmarks.csv    {len(benchmarks):>8,} rows")
    print(f"\n  A/B demo check:")
    ab_high = metrics[metrics["placement_id"].isin(
        placements[placements["campaign_id"] == "CAMP_AB_HIGH"]["placement_id"]
    )]["completion_rate"].mean()
    ab_low = metrics[metrics["placement_id"].isin(
        placements[placements["campaign_id"] == "CAMP_AB_LOW"]["placement_id"]
    )]["completion_rate"].mean()
    print(f"    CAMP_AB_HIGH mean completion rate: {ab_high:.4f} (target ≈ 0.71)")
    print(f"    CAMP_AB_LOW  mean completion rate: {ab_low:.4f} (target ≈ 0.63)")
    print(f"    Gap: {ab_high - ab_low:.4f} (needs to be > 0.05 for significant t-test)")
    print("=" * 60)


if __name__ == "__main__":
    main()
