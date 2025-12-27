# 🧹 Automatic Data Cleaner

A **Python Streamlit app** for preprocessing, cleaning, and transforming datasets, with support for CSV, Excel, and SQL Server sources. Ideal for fast exploratory data cleaning before analytics or modeling.

---

## Features

- **Upload & Load Data**
  - Upload CSV or Excel files.
  - Connect to SQL Server and load tables directly.

- **Data Cleaning Options**
  - Remove duplicates (after normalization).
  - Standardize text fields (trim, title case, email validation).
  - Standardize date fields.

- **Column Transformations**
  - Uppercase, lowercase, title case.
  - Strip whitespace.
  - Remove special characters.

- **Conditional Fills**
  - Fill missing values globally or conditionally based on other columns.

- **Find & Replace**
  - Apply find-and-replace operations on a specific column or the entire dataset.

- **SQL Server Integration**
  - Save cleaned datasets back to SQL Server.
  - Backup tables before overwriting.

- **Preview Mode**
  - Preview cleaned data without saving.
  - Control the number of rows shown.

- **Download Option**
  - Download cleaned CSV when not in preview mode.

---

## Tech Stack

- **Python 3**
- **Streamlit** – Web UI
- **Pandas & NumPy** – Data manipulation
- **pyodbc & SQLAlchemy** – SQL Server integration
- **dateutil** – Flexible date parsing

---

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/themthobisikhoza/automatic-data-cleaner.git
   cd automatic-data-cleaner
