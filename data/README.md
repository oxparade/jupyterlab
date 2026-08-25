# MLOps Dataset

## Overview

This directory contains all resources related to the reference dataset used throughout the ENI MLOps courses.
The dataset serves as the common use case for demonstrations, labs and assessments across multiple modules and courses.
Using a single dataset ensures consistency and allows students to progressively build an end-to-end MLOps workflow.

---

## Dataset Information

### Name

ElectricityLoadDiagrams20112014

### Source

Trindade, A. (2015). ElectricityLoadDiagrams20112014 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86.

### Description

The dataset contains electricity consumption measurements collected from Portuguese consumers.

Measurements are recorded every 15 minutes over approximately four years.

#### Consumer Profiles

The dataset contains 370 Portuguese electricity consumers connected to the medium-voltage network.

The dataset documentation does not provide detailed customer metadata. However, the published literature indicates that the dataset represents aggregated consumption from medium-voltage consumers rather than residential households.

As a result, consumption levels may vary significantly across customers and can reach values that would be unrealistic for residential users.

See:
Ömer Faruk Ertugrul,
Forecasting electricity load by a novel recurrent extreme learning machines approach,
International Journal of Electrical Power & Energy Systems, Volume 78, 2016, Pages 429-435, ISSN 0142-0615,
https://doi.org/10.1016/j.ijepes.2015.12.006.

### Key Characteristics

- 370 consumers (individuals)
- 15-minute granularity
- Approximately 4 years of historical data
- Time-series dataset
- Real-world consumption patterns
- Seasonal effects and concept drift opportunities

### Notable Challenges

- Daylight Saving Time (DST)
- Leap year handling
- Missing values
- Time-series feature engineering
- Data drift between seasons
- Multiple consumption profiles

---

## Directory Structure

```text
dataset/
├── README.md
├── raw/
├── processed/
├── external/
└── metadata/ 
```

### raw/

Original dataset files.

No manual modifications should be performed in this directory.

### processed/

Prepared datasets generated from preprocessing pipelines.

### metadata/

Documentation and dataset-related metadata.

---

## Dataset Versioning with DVC

### Current Status

DVC is initialized in this repository.

However, no shared DVC remote is currently available.

As a result:

- Dataset files are not stored in Git.
- Dataset files are not shared automatically.
- Each instructor manages a local dataset copy.

### Local Setup

Download the dataset manually, rename it in "raw.txt" and place it in:

```text
dataset/raw/
```

### Tracking with DVC

Example:
```bash
dvc add dataset/raw
git add dataset/raw.dvc
git commit -m "Track raw dataset with DVC" 
```


### Shared Remote (Future)

When a shared storage solution becomes available (MinIO, Garage, S3, etc.), a DVC remote can be configured.

Example:
bash 
```bash
dvc remote add -d eni-storage s3://mlops-datasets
dvc push
```

This will allow:

- Shared datasets across instructors
- Reproducible environments
- Dataset synchronization using dvc pull
