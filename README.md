# ⛽ Material Balance Simulator — Oil & Gas Reservoirs

A professional Python-based reservoir simulation tool for estimating **OOIP** (Original Oil-in-Place) and **OGIP** (Original Gas-in-Place) using classical Material Balance Equations (MBE). Built with a premium dark Streamlit UI designed for petroleum engineering professionals.

**Developed by [Mohit Choudhary](https://www.linkedin.com/in/mohit-choudhary-25165730a)**

---

## 🚀 Live App

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mbsimulator-nmnf6eptwgkvcqg2qkikvn.streamlit.app/)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🛢️ **Oil Reservoir MBE** | Full material balance with bubble point, Bo, Rs, Bg, Bw, viscosity |
| 💨 **Gas Reservoir MBE** | Z-factor, Bg, p/Z analysis, OGIP estimation |
| 📊 **Drive Mechanism Analysis** | Stacked area chart showing % contribution of each drive |
| 🔮 **Future Performance** | Pressure & production forecasting beyond historical data |
| 📈 **Campbell & Cole Plots** | Diagnostic plots for reservoir characterization |
| 💧 **Water Influx Modeling** | Pot aquifer model for We calculation |
| 🧪 **Fluid PVT Properties** | Full property curves vs pressure |
| 📤 **Export Results** | Download combined historical + forecast data as CSV |

---

## 🧮 Correlations Used

- **Bo / Rs**: Standing's correlation (Vasquez-Beggs above Pb)
- **Oil Compressibility**: Vasquez-Beggs
- **Z-factor**: Hall-Yarborough (Newton-Raphson iteration)
- **Bw**: McCain's correlation
- **Gas Viscosity**: Lee et al. correlation
- **Oil Viscosity**: Beal's dead oil + Chew-Connally

---

## 📂 Input Data Format

### Oil Reservoir (Excel)
| Column | Unit |
|---|---|
| Date | YYYY-MM-DD |
| Pressure | psi |
| Cum Oil Production | MMSTB |
| Cum Gas Production | MMSCF |
| Cum Water Production | MMSTB |

### Gas Reservoir (Excel)
| Column | Unit |
|---|---|
| Date | YYYY-MM-DD |
| Pressure | psi |
| Cum Gas Production | **BSCF** |
| Cum Water Production | MMSTB (optional) |

---

## ⚙️ Run Locally

```bash
git clone https://github.com/lolamlol1/MBE-simulator-mohit-project-.git
cd MBE-simulator-mohit-project-
pip install -r requirements.txt
streamlit run MaterialBalanceSimulator.py
```

---

## 📦 Requirements

```
streamlit
matplotlib
pandas
numpy
scipy
scikit-learn
openpyxl
xlrd>=2.0.1
XlsxWriter
```

---

## 👤 Author

**Mohit Choudhary**
🔗 [LinkedIn](https://www.linkedin.com/in/mohit-choudhary-25165730a)

---

> *Built for petroleum engineering professionals — OOIP/OGIP estimation, drive mechanism analysis, and production forecasting in one tool.*
