# README

## Insurance Charges Prediction

This project implements a predictive model for insurance charges using various machine learning algorithms. The aim is to analyze and predict the insurance charges based on various features such as age, sex, BMI, number of children, smoking status, and region.

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Modeling](#modeling)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

### Installation

To run this project, ensure you have Python 3.x installed along with the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/insurance-charges-prediction.git
   cd insurance-charges-prediction
   ```

2. Place the `insurance.csv` dataset in the same directory as the script.

3. Run the script:
   ```bash
   python your_script_name.py
   ```

### Data

The dataset used is `insurance.csv`, which contains the following columns:
- **age**: Age of the insured individual
- **sex**: Gender of the insured
- **bmi**: Body Mass Index
- **children**: Number of children/dependents covered by the insurance
- **smoker**: Whether the insured is a smoker or not
- **region**: The region of the insured
- **charges**: The insurance charges

### Modeling

The following machine learning models are implemented:
- Support Vector Regression (SVR)
- Ridge Regression
- Lasso Regression
- Gradient Boosting Regressor (GBR)

Each model is evaluated using Mean Squared Error (MSE) and R² Score.

### Results

The performance metrics for each model are saved in a `summary.json` file, which includes:
- R² Score
- Mean Squared Error (MSE)
- Accuracy Score

### Visualization

Various visualizations are created to analyze the data and model performance:
- Heatmap of feature correlations
- Distribution plots
- Bar plots comparing charges by age, region, and number of children
- Box plots for smokers and non-smokers
- Scatter plots for BMI versus charges
- Bar plots for model evaluation metrics (R² Score, Accuracy, MSE)

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to contribute to this project by opening issues or submitting pull requests!
