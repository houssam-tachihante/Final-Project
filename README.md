# Supply Chain Analytics Dashboard
**Machine Learning & Deep Learning Applications in Supply Chain Management**

## Project Overview
This project presents an interactive Streamlit dashboard that applies Machine Learning (ML), Deep Learning (DL), and Data Analytics techniques to a real-world supply chain dataset.
The dashboard is designed to empirically explore and validate concepts from two recent academic research papers in supply chain analytics.
The project integrates:
- Predictive analytics
- Clustering & segmentation
- Interpretability via feature importance
- Practical supply chain KPIs

using the DataCo Smart Supply Chain Dataset.

## Dataset

**Dataset Name:** DataCo Smart Supply Chain Dataset
**Source:** Kaggle / Academic use
**Observations:** ~180,000
**Features:** 53
**Key Dimensions:**
- Orders & shipments
- Products & categories
- Customers & regions
- Sales, profit, and delivery risk

This dataset supports demand forecasting, logistics analysis, customer segmentation, and inventory analytics.

## Research Papers Used

### Research Paper 

**Title:**
Data Analytics in Supply Chain Management: A State-of-the-Art Literature Review

**Authors:**
Farzaneh Darbanian, Patrick Brandtner, Taha Falatouri, Mehran Nasseri

**Journal:**
Operations and Supply Chain Management (OSCM), Vol. 17, No. 1, 2024

**Key Contribution:**
This paper presents a systematic literature review of 354 studies, categorizing:
- Supply chain functions
- Analytics maturity levels (descriptive → prescriptive)
- Common data analytics techniques
- Emerging research gaps

It provides a roadmap for applying analytics across demand, logistics, inventory, and risk management.

## Research Questions Addressed
The dashboard answers four core research questions, inspired by and aligned with both papers:

**RQ1 – Key Performance Drivers**
Which factors most influence key supply chain performance metrics?
- Feature importance via Random Forest models
- Interpretable ML insights (Paper 1 focus)

**RQ2 – ML vs Deep Learning**
Can deep learning models outperform classical ML in predicting delivery delays or order risks?
- Random Forest vs Neural Network comparison
- Metric-based performance evaluation

**RQ3 – Dimensionality Reduction & Interpretability**
How can dimensionality reduction improve interpretability and accuracy?
- PCA-based visualization
- Feature-space simplification (Paper 1 & Paper 2 alignment)

**RQ4 – Clustering & Segmentation**
Can customers or products be clustered into meaningful behavioral groups?
- KMeans clustering
- PCA-based cluster visualization
- Inventory & demand risk insights (Paper 2 focus)

```
## Tools & Technologies
- Python 3.10+
- Pandas & NumPy – Data processing
- Scikit-learn – ML pipelines, clustering, PCA
- TensorFlow / Keras – Neural networks
- Streamlit – Interactive dashboard
- Altair / Matplotlib – Visualization

## How to Run the Dashboard
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Paper dashboard
```bash
streamlit run ResearchPaper.py
```

## Academic Value

This project:
- Translates theoretical research into applied analytics
- Demonstrates ML/DL pipelines on real supply chain data
- Emphasizes interpretability, not just accuracy
- Bridges literature review insights with practical modeling



