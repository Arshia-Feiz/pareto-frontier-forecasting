# Two-Dimensional Pareto Frontier Forecasting for Technology Planning

## 📊 Project Overview

This project replicates the methodology from the research paper "Two-dimensional Pareto frontier forecasting for technology planning and roadmapping" by Yuskevich, Vingerhoeds & Golkar. The work focuses on identifying and modeling Pareto frontiers in automotive technology, specifically analyzing the trade-off between horsepower and fuel efficiency.

## 🎯 Key Objectives

- **Pareto Frontier Identification**: Find optimal trade-offs between competing objectives
- **Technology Roadmapping**: Model efficiency frontiers for strategic planning
- **Optimization Analysis**: Understand performance vs. efficiency trade-offs
- **Methodological Validation**: Recreate academic research methodology

## 🛠️ Technical Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations and optimization
- **Matplotlib/Seaborn**: Data visualization
- **Scipy**: Optimization algorithms
- **Jupyter Notebooks**: Interactive analysis

## 📈 Methodology

### 1. Data Preprocessing
- **Dataset**: Kaggle car specifications (1970-2015)
- **Filtering**: Petrol/manual transmission vehicles only
- **Unit Conversion**: L/100km to MPG for fuel efficiency
- **Quality Control**: Removed outliers and incomplete records

### 2. Pareto Frontier Analysis
- **Objective Functions**: Maximize horsepower while maximizing fuel efficiency (MPG)
- **Frontier Identification**: Custom algorithm to extract optimal points
- **Curve Fitting**: Logarithmic approximation of Pareto frontier
- **Visualization**: Color-mapped scatter plots with frontier overlay

### 3. Validation and Comparison
- **Methodological Reproduction**: Recreate original paper's approach
- **Visual Comparison**: Compare results with original research
- **Statistical Validation**: Assess frontier approximation accuracy
- **Robustness Testing**: Validate across different time periods

## 📊 Results

### Key Visualizations
- **Pareto Frontier Plots**: Horsepower vs. MPG with optimal frontier
- **Temporal Evolution**: How efficiency frontiers change over time
- **Manufacturer Analysis**: Different companies' positioning relative to frontier
- **Technology Trends**: Evolution of performance-efficiency trade-offs

### Analytical Insights
- **Efficiency Trade-offs**: Clear relationship between power and efficiency
- **Technology Progression**: Frontier shifts over time indicating improvements
- **Manufacturer Strategies**: Different approaches to frontier optimization
- **Strategic Implications**: Understanding optimal technology positioning

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### Data Requirements
- Kaggle car specifications dataset (1970-2015)
- Petrol/manual transmission vehicles
- Horsepower and fuel efficiency data
- Temporal information for trend analysis

### Pareto Frontier Analysis
```python
# Example Pareto frontier identification
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def identify_pareto_frontier(horsepower, mpg):
    """
    Identify Pareto frontier points
    """
    # Sort by one objective and find non-dominated solutions
    sorted_indices = np.argsort(horsepower)
    pareto_points = []
    
    for i in sorted_indices:
        is_dominated = False
        for j in pareto_points:
            if (horsepower[j] >= horsepower[i] and mpg[j] >= mpg[i] and 
                (horsepower[j] > horsepower[i] or mpg[j] > mpg[i])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(i)
    
    return pareto_points
```

## 📁 Project Structure

```
pareto-frontier-forecasting/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── car_specifications.csv
│   └── processed/
│       ├── filtered_data.csv
│       └── pareto_points.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pareto_analysis.ipynb
│   ├── 03_frontier_modeling.ipynb
│   └── 04_validation_comparison.ipynb
├── src/
│   ├── data_processing.py
│   ├── pareto_analysis.py
│   ├── frontier_modeling.py
│   └── visualization.py
├── results/
│   ├── visualizations/
│   │   ├── pareto_frontier.png
│   │   ├── temporal_evolution.png
│   │   └── manufacturer_analysis.png
│   └── models/
│       └── frontier_approximation.pkl
├── documentation/
│   └── (PDF available via Google Drive link)
└── documentation/
    └── methodology_notes.md
```

## 📄 Documentation

This project includes comprehensive PDF documentation that provides:
- **Pareto Frontier Methodology**: Complete implementation guide
- **Multi-Objective Optimization**: Detailed algorithm explanations
- **Frontier Identification**: Custom algorithm development
- **Validation Procedures**: Comparison with original research

**📖 View Documentation**: [Recreating_Two_dimensional_Pareto.pdf](https://drive.google.com/file/d/1_4QTaAt04O1NUFitQg6WTrRsM9dxKvjG/view?usp=share_link)

## 🔬 Research Applications

This project demonstrates:
- **Multi-Objective Optimization**: Balancing competing objectives
- **Technology Strategy**: Understanding efficiency frontiers
- **Strategic Planning**: Supporting technology roadmapping decisions
- **Academic Research**: Validating published methodologies

## 📚 References

- **Original Paper**: Yuskevich, Vingerhoeds & Golkar "Two-dimensional Pareto frontier forecasting"
- **Dataset**: [Kaggle Car Specifications](https://www.kaggle.com/datasets/CooperUnion/car-dataset)
- **Multi-Objective Optimization**: Pareto frontier analysis techniques

## 🎓 Academic Context

This work was conducted at ISAE-SUPAERO as part of research into:
- Technology roadmapping methodologies
- Multi-objective optimization in technology planning
- Efficiency frontier analysis
- Strategic technology positioning

## 👨‍💻 Author

**Arshia Feizmohammady**
- Industrial Engineering Student, University of Toronto
- Research focus: Technology optimization and strategic planning
- [LinkedIn](https://linkedin.com/in/arshiafeiz)
- [Personal Website](https://arshiafeizmohammady.com)

## 📄 License

This project is for educational and research purposes. Please cite the original paper and this reproduction appropriately.

---

*This project reproduces academic research methodology for educational purposes and demonstrates the application of Pareto frontier analysis in technology planning.*
