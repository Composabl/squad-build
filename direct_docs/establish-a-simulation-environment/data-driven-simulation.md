# Data Driven Simulation

## Data-Driven Simulations

AMESA enables you to build simulations from historical data. These simulations are used to train and evaluate multi-agent systems in high-fidelity environments that reflect your actual operations.

Use cases include modeling equipment behavior, logistics workflows, and process automation.

### Recommended Data Range

* **Minimum**: 3 months
* **Preferred**: 1 year (especially for systems with seasonal variation)

Datasets will likely yield a successful simulation when they reflect a >80% accurate relationship between control actions (operator adjustments) and sensor readings.

***

### 1. CSV Data Format

Upload a CSV file where each column follows a naming convention:

| Variable Type | Prefix        | Example                 |
| ------------- | ------------- | ----------------------- |
| Sensor        | `s_`          | `s_T` (temperature)     |
| Action        | `a_`          | `a_dTc` (cooling delta) |
| Other Input   | _(no prefix)_ | `Datetime`              |

**Sensors** are variables that provide information about the environment or conditions within the process. These may be reported by the machine being controlled or they may come from outside systems. Quality measures such as the results of lab tests are also sensors.

**Actions** are variables that describe the adjustments the operator makes to the system controls.

> Optionally include a **timestamp** column in UTC. If omitted, AMESA assumes the rows are sequential time steps.
>
> Not all fields need values for every row. The simulator recognizes when data is collected at different intervals.

#### Units Row (Optional)

Include a row above your column headers to define units for each variable:

```csv
kmol/m3,K,K,,kmol/m3,K
s_Ca,s_T,s_Tc,s_Tref,s_Cref,a_dTc
…data rows…
```

### 2. Data Configuration

After uploading your CSV file, AMESA guides you through a configuration screen where you confirm the role and type of each column:

| **Field**          | **Description**                               |
| ------------------ | --------------------------------------------- |
| User variable name | Rename your variables for clarity (optional)  |
| Mode               | Sensor / Action / None (for reference values) |
| Type               | Box or continuous variables                   |
| Value Range        | Auto-calculated from your data                |
| Unit (optional)    | Populated from the units row                  |

***

### 3. Simulation Creation Flow

#### 🛠 Step 1: Upload CSV

* Navigate to the Data-Driven Simulator section.
* Name your simulation.
* Upload your CSV file in the proper format.

<figure><img src="../.gitbook/assets/Upload File.png" alt=""><figcaption></figcaption></figure>

#### 🧭 Step 2: Configure Variables

* Review detected variables.
* Adjust mode, type, unit, and value range if needed.

<figure><img src="../.gitbook/assets/Review and update simulation.png" alt=""><figcaption></figcaption></figure>

#### 📊 Step 3: Review Data Score

AMESA will scan your data and generate a Data Score from 0 to 100:

| **Score Range** | **Meaning**                                   |
| --------------- | --------------------------------------------- |
| 80–100          | Good quality – Ready to simulate              |
| Below 80        | Needs improvement – Add data or clean inputs. |

<figure><img src="../.gitbook/assets/Data Quality Score.png" alt=""><figcaption></figcaption></figure>

#### 🚀 Step 4: Create Simulation

Click Next, and your simulation will be created. It will now be available in your workspace for training, testing, and analysis.

<figure><img src="../.gitbook/assets/Building the simulation.png" alt=""><figcaption></figcaption></figure>

***

### &#x20;4. Best Practices

* Use consistent naming: Prefix sensors and actions properly.
* Include a units row: This improves interpretability and ensures correctness.
* Handle missing data: Some missing values are fine, but complete coverage of key variables improves accuracy.
* Timestamps (optional): Include UTC timestamps if events need temporal alignment.

***

### 5. Example File

📁 Download sample data:

{% file src="../.gitbook/assets/cstr_simulator_data 2.csv" %}

ple.csv

Use this example file, which contains data, to create a [Chemical Process](../tutorials/industrial-mixer/) Simulation and see the expected structure, naming conventions, and units in action.

***

### 6. Troubleshooting

| **Problem**           | **Cause**                       | **Fix**                                         |
| --------------------- | ------------------------------- | ----------------------------------------------- |
| Variable not detected | Missing prefix                  | Use s\_ for sensors and a\_ for actions         |
| Low data score        | Not enough data, missing values | Add more historical data or fill in key columns |
| Unit not recognized   | Unit row missing or misaligned  | Add units row directly above the variable row   |
