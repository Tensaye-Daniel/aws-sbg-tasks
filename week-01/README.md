# Week 01 - Student Data Processing

**Objective**

Load a raw student dataset, clean it, compute summary statistics, classify every
student as `Pass` or `Fail`, and export the cleaned dataset.

**What the task covers**

| Step | What it does | Tool |
| --- | --- | --- |
| 1. Load & inspect | Read the CSV, check dtypes and missing values | `pandas` |
| 2. Clean | Remove duplicates, impute missing values | `pandas` |
| 3. Statistics | Average, max, min, standard deviation of scores | `numpy` |
| 4. Classify | Add a `Pass` / `Fail` column (pass mark = 50) | `numpy` |
| 5. Save | Write `data/cleaned_students.csv` | `pandas` |

**Files**

- `student_data_processing.py` - the full pipeline as a standalone script.
- `student_data_processing.ipynb` - the same pipeline as a Google Colab
  notebook, with outputs and explanations included.
- `data/students.csv` - the raw input dataset.
- `data/cleaned_students.csv` - the processed output.
- `requirements.txt` - the packages needed to run the code.

**How to run**

```bash
pip install -r requirements.txt
python student_data_processing.py
```

The script prints the raw data, the cleaned data, the score statistics and the
final classified dataset, then writes `data/cleaned_students.csv`.

**Google Colab**

Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive1uRvsuRbjnZXanE_Y0LhE3-SuwIyCfUUJ?usp=sharing)

The notebook loads `data/students.csv` from the local `data/` folder, falls back
to the raw GitHub URL of this repository, and finally to an embedded copy of the
data, so it runs end to end either way.

The committed `.ipynb` already contains the executed code, outputs and written
explanations, so it can be reviewed on GitHub without running anything.

**Notes on the cleaning decisions**

- A missing **age** is filled with the **median** age - ages are spread out, so
  the median is less affected by outliers than the mean would be.
- A missing **score** is filled with the **mean** score.
- Cleaning functions each return a **new** DataFrame, so the raw data is never
  modified in place.
