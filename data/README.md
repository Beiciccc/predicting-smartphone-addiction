# Data notes

Download the competition files from Kaggle:

```bash
kaggle competitions download -c playground-series-s6e8
unzip playground-series-s6e8.zip
```

The raw CSV files and downloaded archives are intentionally excluded from this repository.

## Schema

| Column | Role | Type |
|---|---|---|
| `id` | row identifier | integer |
| `age` | feature | numeric |
| `daily_screen_time_hours` | feature | numeric |
| `social_media_hours` | feature | numeric |
| `gaming_hours` | feature | numeric |
| `work_study_hours` | feature | numeric |
| `sleep_hours` | feature | numeric |
| `notifications_per_day` | feature | numeric |
| `app_opens_per_day` | feature | numeric |
| `weekend_screen_time` | feature | numeric |
| `gender` | feature | categorical |
| `stress_level` | feature | categorical |
| `academic_work_impact` | feature | categorical |
| `addicted_label` | training target | binary integer |

## Aggregate checks

- Training shape: 691,369 rows × 14 columns.
- Test shape: 296,302 rows × 13 columns.
- Training positive rate: 70.9424%.
- IDs are unique and disjoint across train and test.
- No exact duplicate feature rows were found within either split.
- All twelve features contain missing values.
- Observed numeric-value distributions align closely between train and test, while column-level missingness rates differ.

These figures describe the August 3, 2026 competition download.
