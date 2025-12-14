# ResearchPaper2.py
# Streamlit app for Research Paper 2:
# "Data Analytics in Supply Chain Management: A State-of-the-Art Literature Review" (2024)

import streamlit as st
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import altair as alt

# -----------------------------
# Basic page config
# -----------------------------
st.set_page_config(
    page_title="Supply Chain Data Analytics – Literature Review Dashboard",
    layout="wide"
)

# -----------------------------
# Data loading helper
# -----------------------------
@st.cache_data
def load_data():
    # Adjust path if needed; encoding matches DataCo dataset
    df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="ISO-8859-1")
    return df


df = load_data()

# -----------------------------
# SIDEBAR: Main Navigation
# -----------------------------
st.sidebar.title("Navigation – Paper 2")

page = st.sidebar.radio(
    "Go to",
    (
        "Home",
        "Data Overview",
        "RQ1 – Demand Forecasting",
        "RQ2 – Customer / Product Segmentation",
        "RQ3 – Late Delivery Prediction",
        "RQ4 – Inventory & Stock Risk Insights",
        "About / Methods (Paper 2)"
    )
)

st.sidebar.markdown("---")
st.sidebar.write("Dataset shape:")
st.sidebar.code(df.shape)

# -----------------------------
# PAGE ROUTING
# -----------------------------
if page == "Home":
    st.title("📦 Supply Chain Data Analytics – Literature Review Perspective")

    st.markdown("""
    ## 📝 Research Paper Reference (Paper 2)

    **Title:** Data Analytics in Supply Chain Management: A State-of-the-Art Literature Review  
    **Authors:** Farzaneh Darbanian, Patrick Brandtner, Taha Falatouri, Mehran Nasseri  
    **Journal:** Operations and Supply Chain Management (OSCM), Vol. 17, No. 1, 2024  
    **Type:** Systematic Literature Review (354 papers, 2020–2021)  
    """)

    st.markdown("""
    ## 📘 Summary of the Paper

    This paper conducts a **systematic review of 354 research articles** on
    **Data Analytics in Supply Chain Management (SCM)**. It classifies the literature by:

    - **Supply chain functions:** demand management, manufacturing, logistics,
      warehousing, procurement, risk management, sustainability, etc.  
    - **Analytics levels:** descriptive, diagnostic, predictive, prescriptive.  
    - **Techniques used:** statistics, machine learning, optimization, simulation,
      hybrid / mixed approaches.  

    ### Key Insights from the Review

    - There is a strong **growth** in data analytics applications in SCM, especially in
      **demand forecasting, logistics, and manufacturing**.
    - **Predictive analytics** is now the most frequently used analytics level.
    - **Hybrid models** (combining ML, optimization, simulation) are becoming more common.
    - Important **research gaps** include:
      - Real-time and streaming analytics in supply chains  
      - Integration of external / public data sources  
      - Advanced analytics for procurement, order picking, in-transit inventory,
        and demand shaping under uncertainty  

    This dashboard uses the **DataCoSupplyChainDataset** to build concrete examples of
    how data analytics can answer practical SCM questions that are highlighted in the review.
    """)

    st.markdown("---")

    st.markdown("""
    ## 🎯 Research Questions Addressed in This Dashboard (Paper 2 View)

    These four questions are designed to align with common themes and gaps found in the
    literature review, but implemented using your **DataCoSupplyChainDataset**.

    ### 🔹 RQ1 – Demand Forecasting
    **Question:**  
    *How can predictive analytics improve demand forecasting accuracy for different product
    categories, regions, or customer segments?*

    ### 🔹 RQ2 – Customer / Product Segmentation
    **Question:**  
    *What customer or product segments can be identified using clustering to support
    marketing and operational decisions?*

    ### 🔹 RQ3 – Late Delivery Prediction
    **Question:**  
    *Which factors influence late deliveries, and how well can we predict late-delivery risk
    using machine learning models?*

    ### 🔹 RQ4 – Inventory & Stock Risk Insights
    **Question:**  
    *What inventory-related patterns (e.g., high demand variability, low margins) can be
    detected that indicate stock-out risk or excess inventory?*

    Explore each RQ using the navigation menu on the left.
    """)

    st.info("Tip: Start with **Data Overview** to get familiar with the dataset used in all four RQs.")

# ---------------------------------------------------------------------
elif page == "Data Overview":
    st.title("📊 Data Overview – DataCoSupplyChainDataset")

    st.write("Below is a quick look at the dataset used for all analyses in this Paper 2 view.")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Shape")
        st.write(f"Rows: **{df.shape[0]:,}**")
        st.write(f"Columns: **{df.shape[1]}**")

        st.subheader("Basic Info")
        st.write("Numeric columns:")
        st.write(df.select_dtypes(include=["int64", "float64"]).columns.tolist()[:20])

        st.write("Categorical columns:")
        st.write(df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()[:20])

    with col2:
        st.subheader("Descriptive Statistics (Numeric)")
        st.write(df.describe())

    st.markdown("---")
    st.subheader("Column Names")
    st.code(", ".join(df.columns))

# ---------------------------------------------------------------------
elif page == "RQ1 – Demand Forecasting":
    st.title("RQ1 – Demand Forecasting with Predictive Analytics")

    st.markdown("""
    ### Research Question  
    *How can predictive analytics improve demand forecasting accuracy for different product
    categories, regions, or customer segments?*

    We treat **demand** as a numeric target such as order quantity or sales and use a
    RandomForest regression model to learn how features (product, customer, region, dates)
    explain this demand.
    """)

    # -----------------------------
    # 1. Choose target (demand proxy)
    # -----------------------------
    st.subheader("1. Select Demand Target")

    candidate_targets = [
        "Order Item Quantity",
        "Sales",
        "Order Item Total"
    ]
    available_targets = [c for c in candidate_targets if c in df.columns]

    if not available_targets:
        st.error("No suitable numeric demand columns found in the dataset.")
        st.stop()

    target_col = st.selectbox(
        "Select a demand-related target variable:",
        available_targets
    )

    st.write(f"**Selected target:** `{target_col}`")

    # -----------------------------
    # 2. Prepare data & sample for speed
    # -----------------------------
    st.subheader("2. Data Preparation & Sampling")

    data = df.dropna(subset=[target_col]).copy()

    # Optional: simple date features if available
    if "order date (DateOrders)" in data.columns:
        data["order_date_parsed"] = pd.to_datetime(
            data["order date (DateOrders)"], errors="coerce"
        )
        data["order_year"] = data["order_date_parsed"].dt.year
        data["order_month"] = data["order_date_parsed"].dt.month
        data["order_dow"] = data["order_date_parsed"].dt.dayofweek

    X = data.drop(columns=[target_col])
    y = data[target_col].astype(float)

    max_rows = st.slider(
        "Maximum rows to use for training (sampling):",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )

    original_len = len(X)
    if original_len > max_rows:
        sample = data.sample(n=max_rows, random_state=42)
        X = sample.drop(columns=[target_col])
        y = sample[target_col].astype(float)
        st.info(f"Using a sample of **{max_rows:,} rows** (from {original_len:,}).")
    else:
        st.info(f"Using all **{original_len:,} rows**.")

    test_size = st.slider(
        "Test set size (fraction):",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    # -----------------------------
    # 3. Preprocessing
    # -----------------------------
    st.subheader("3. Preprocessing & Model Setup")

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    st.markdown("**Numeric features (sample):** " + ", ".join(numeric_features[:8]) +
                (" ..." if len(numeric_features) > 8 else ""))
    st.markdown("**Categorical features (sample):** " + ", ".join(categorical_features[:8]) +
                (" ..." if len(categorical_features) > 8 else ""))

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # -----------------------------
    # 4. Train model
    # -----------------------------
    st.subheader("4. Train Model & Evaluate")

    if st.button("Train Demand Model"):
        with st.spinner("Training RandomForest regressor..."):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

        st.success("Model trained!")

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.markdown("### 📊 Performance Metrics")
        st.write(f"**MAE:** {mae:.3f}")
        st.write(f"**R²:** {r2:.3f}")

        # -------------------------
        # 5. Feature importances
        # -------------------------
        st.markdown("### 🔍 Feature Importances")

        rf = pipe.named_steps["model"]
        pre = pipe.named_steps["preprocess"]

        try:
            feature_names = pre.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(len(rf.feature_importances_))]

        importances = rf.feature_importances_
        n = min(len(feature_names), len(importances))
        fi_df = pd.DataFrame({
            "Feature": feature_names[:n],
            "Importance": importances[:n]
        }).sort_values("Importance", ascending=False)

        top_n = st.slider("Show top N features:", 5, 30, 15)
        st.dataframe(fi_df.head(top_n))

        st.bar_chart(fi_df.head(top_n).set_index("Feature")["Importance"])

        st.info(
            "These features are the most important drivers of the selected demand target "
            "according to the RandomForest model."
        )
    else:
        st.warning("Click **Train Demand Model** to run the forecasting analysis.")

# ---------------------------------------------------------------------
elif page == "RQ2 – Customer / Product Segmentation":
    st.title("RQ2 – Customer / Product Segmentation via Clustering")

    st.markdown("""
    ### Research Question  
    *What customer or product segments can be identified using clustering to support
    strategic decision-making?*

    We use **KMeans clustering** on selected features and then visualize the clusters
    in a 2D PCA space.
    """)

    # -----------------------------
    # 1. Select features for clustering
    # -----------------------------
    st.subheader("1. Select Features")

    data = df.copy()

    # Identify numeric and categorical features
    numeric_features_all = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features_all = data.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    st.markdown("**Numeric features (sample):** " + ", ".join(numeric_features_all[:8]) +
                (" ..." if len(numeric_features_all) > 8 else ""))
    st.markdown("**Categorical features (sample):** " + ", ".join(categorical_features_all[:8]) +
                (" ..." if len(categorical_features_all) > 8 else ""))

    default_num = [c for c in numeric_features_all if c in [
        "Order Item Quantity", "Sales", "Order Item Total", "Order Profit Per Order"
    ]]
    if not default_num:
        default_num = numeric_features_all[:5]

    default_cat = [c for c in categorical_features_all if c in [
        "Customer Country", "Customer Segment", "Order Region", "Category Name"
    ]]
    if not default_cat:
        default_cat = categorical_features_all[:5]

    selected_numeric = st.multiselect(
        "Numeric features:",
        options=numeric_features_all,
        default=default_num
    )

    selected_categorical = st.multiselect(
        "Categorical features:",
        options=categorical_features_all,
        default=default_cat
    )

    if not selected_numeric and not selected_categorical:
        st.error("Please select at least one numeric or categorical feature.")
        st.stop()

    X = data[selected_numeric + selected_categorical].copy()

    # -----------------------------
    # 2. Sampling & preprocessing
    # -----------------------------
    st.subheader("2. Sampling & Preprocessing")

    max_rows = st.slider(
        "Maximum rows to cluster:",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )

    original_len = len(X)
    if original_len > max_rows:
        X = X.sample(n=max_rows, random_state=42)
        st.info(f"Using **{max_rows:,} rows** (from {original_len:,}).")
    else:
        st.info(f"Using all **{original_len:,} rows**.")

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    n_clusters = st.slider(
        "Number of clusters (K):",
        min_value=2,
        max_value=10,
        value=4,
        step=1
    )

    # -----------------------------
    # 3. Run clustering
    # -----------------------------
    st.subheader("3. Run KMeans Clustering")

    if st.button("Run Segmentation"):
        with st.spinner("Clustering..."):
            X_proc = preprocessor.fit_transform(X)
            X_proc = np.asarray(X_proc, dtype="float32")

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init="auto"
            )
            labels = kmeans.fit_predict(X_proc)

            X_clustered = X.copy()
            X_clustered["Cluster"] = labels

            # Cluster sizes
            st.markdown("### 📊 Cluster Sizes")
            counts = X_clustered["Cluster"].value_counts().sort_index()
            size_df = pd.DataFrame({
                "Cluster": counts.index,
                "Count": counts.values
            })
            st.dataframe(size_df)
            st.bar_chart(size_df.set_index("Cluster")["Count"])

            # Numeric feature means
            if numeric_features:
                st.markdown("### 🧬 Cluster Profiles (Numeric Means)")
                profile_df = X_clustered.groupby("Cluster")[numeric_features].mean().round(2)
                st.dataframe(profile_df)

            # PCA 2D visualization
            st.markdown("### 🌐 2D PCA Visualization of Clusters")
            if X_proc.shape[1] >= 2:
                pca_2d = PCA(n_components=2)
                X_2d = pca_2d.fit_transform(X_proc)
                vis_df = pd.DataFrame({
                    "PC1": X_2d[:, 0],
                    "PC2": X_2d[:, 1],
                    "Cluster": labels.astype(str)
                })

                chart = (
                    alt.Chart(vis_df)
                    .mark_circle(size=40, opacity=0.6)
                    .encode(
                        x="PC1",
                        y="PC2",
                        color=alt.Color("Cluster", type="nominal"),
                        tooltip=["PC1", "PC2", "Cluster"]
                    )
                    .properties(height=400)
                )
                st.altair_chart(chart, use_container_width=True)

                st.info(
                    "Clusters in this PCA space represent groups of customers/products "
                    "with similar behavior across the selected features."
                )
            else:
                st.warning("Not enough dimensions after preprocessing to compute PCA.")
    else:
        st.warning("Click **Run Segmentation** to compute clusters.")

# ---------------------------------------------------------------------
elif page == "RQ3 – Late Delivery Prediction":
    st.title("RQ3 – Late Delivery Prediction & Risk Factors")

    st.markdown("""
    ### Research Question  
    *Which factors influence late deliveries, and how well can we predict late-delivery risk using
    machine learning models?*

    We use `Late_delivery_risk` as the target and train a RandomForest classifier.
    """)

    if "Late_delivery_risk" not in df.columns:
        st.error("Column `Late_delivery_risk` not found in dataset.")
        st.stop()

    # -----------------------------
    # 1. Prepare data
    # -----------------------------
    st.subheader("1. Data Preparation")

    data = df.dropna(subset=["Late_delivery_risk"]).copy()
    X = data.drop(columns=["Late_delivery_risk"])
    y = data["Late_delivery_risk"]

    max_rows = st.slider(
        "Maximum rows to use:",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )

    original_len = len(X)
    if original_len > max_rows:
        sample = data.sample(n=max_rows, random_state=42)
        X = sample.drop(columns=["Late_delivery_risk"])
        y = sample["Late_delivery_risk"]
        st.info(f"Using **{max_rows:,} rows** (from {original_len:,}).")
    else:
        st.info(f"Using all **{original_len:,} rows**.")

    test_size = st.slider(
        "Test set size (fraction):",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    # -----------------------------
    # 2. Preprocessing & model
    # -----------------------------
    st.subheader("2. Preprocessing & Model")

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # -----------------------------
    # 3. Train model
    # -----------------------------
    st.subheader("3. Train Model & Evaluate")

    if st.button("Train Late-Delivery Model"):
        with st.spinner("Training classifier..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y if y.nunique() > 1 else None
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

        st.success("Model trained!")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        st.markdown("### 📊 Performance Metrics")
        st.write(f"**Accuracy:** {acc:.3f}")
        st.write(f"**F1 (macro):** {f1:.3f}")

        # -------------------------
        # 4. Feature importances
        # -------------------------
        st.markdown("### 🔍 Feature Importances")

        rf = pipe.named_steps["model"]
        pre = pipe.named_steps["preprocess"]

        try:
            feature_names = pre.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(len(rf.feature_importances_))]

        importances = rf.feature_importances_
        n = min(len(feature_names), len(importances))
        fi_df = pd.DataFrame({
            "Feature": feature_names[:n],
            "Importance": importances[:n]
        }).sort_values("Importance", ascending=False)

        top_n = st.slider("Show top N features:", 5, 30, 15)
        st.dataframe(fi_df.head(top_n))
        st.bar_chart(fi_df.head(top_n).set_index("Feature")["Importance"])

        st.info("These features are the most influential for predicting late delivery risk.")
    else:
        st.warning("Click **Train Late-Delivery Model** to run the analysis.")

# ---------------------------------------------------------------------
elif page == "RQ4 – Inventory & Stock Risk Insights":
    st.title("RQ4 – Inventory & Stock Risk Insights")

    st.markdown("""
    ### Research Question  
    *What inventory-related patterns (e.g., high demand variability, low margins) can be detected
    that indicate stock-out risk or excess inventory?*

    We aggregate the data at **product level** and then apply **KMeans clustering** to group
    products with similar demand and profitability characteristics.
    """)

    if not {"Product Name", "Order Item Quantity", "Sales"}.issubset(df.columns):
        st.error("Required columns (`Product Name`, `Order Item Quantity`, `Sales`) not found.")
        st.stop()

    # -----------------------------
    # 1. Aggregate per product
    # -----------------------------
    st.subheader("1. Aggregate Product-Level Metrics")

    inv_df = (
        df.groupby("Product Name")
          .agg(
              total_demand=("Order Item Quantity", "sum"),
              avg_demand=("Order Item Quantity", "mean"),
              demand_std=("Order Item Quantity", "std"),
              avg_sales=("Sales", "mean"),
              total_sales=("Sales", "sum")
          )
          .reset_index()
    )

    st.dataframe(inv_df.head(20))

    # -----------------------------
    # 2. Select features & K
    # -----------------------------
    st.subheader("2. Select Features & Number of Clusters")

    feature_cols = ["total_demand", "avg_demand", "demand_std", "avg_sales", "total_sales"]
    selected_features = st.multiselect(
        "Features for clustering:",
        options=feature_cols,
        default=feature_cols
    )

    if not selected_features:
        st.error("Please select at least one feature for clustering.")
        st.stop()

    X = inv_df[selected_features].fillna(0)

    n_clusters = st.slider(
        "Number of product clusters:",
        min_value=2,
        max_value=10,
        value=4,
        step=1
    )

    # -----------------------------
    # 3. Run clustering
    # -----------------------------
    st.subheader("3. Run KMeans Clustering on Products")

    if st.button("Cluster Products"):
        with st.spinner("Clustering products..."):
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init="auto"
            )
            labels = kmeans.fit_predict(X)

            inv_df["Cluster"] = labels

        st.success("Clustering complete!")

        # Cluster sizes
        st.markdown("### 📊 Cluster Sizes")
        counts = inv_df["Cluster"].value_counts().sort_index()
        size_df = pd.DataFrame({"Cluster": counts.index, "Count": counts.values})
        st.dataframe(size_df)
        st.bar_chart(size_df.set_index("Cluster")["Count"])

        # Cluster profiles
        st.markdown("### 🧬 Cluster Profiles (Averages)")
        profile_df = inv_df.groupby("Cluster")[selected_features].mean().round(2)
        st.dataframe(profile_df)

        # PCA visualization
        st.markdown("### 🌐 2D PCA Visualization of Product Clusters")

        if X.shape[1] >= 2:
            pca_2d = PCA(n_components=2)
            X_2d = pca_2d.fit_transform(X.values)

            # Build visualization dataframe and INCLUDE selected features for tooltip
            vis_df = pd.DataFrame({
                "PC1": X_2d[:, 0],
                "PC2": X_2d[:, 1],
                "Cluster": labels.astype(str),
                "Product Name": inv_df["Product Name"].astype(str)
            })

            # Add selected features into vis_df so Altair can reference them in tooltips
            for col in selected_features:
                vis_df[col] = inv_df[col].values

            chart = (
                alt.Chart(vis_df)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("PC1:Q", title="PC1"),
                    y=alt.Y("PC2:Q", title="PC2"),
                    color=alt.Color("Cluster:N"),
                    tooltip=[
                        alt.Tooltip("Product Name:N"),
                        alt.Tooltip("Cluster:N"),
                        *[alt.Tooltip(f"{c}:Q") for c in selected_features]
                    ],
                )
                .properties(height=400)
            )

            st.altair_chart(chart, use_container_width=True)

            st.info(
                "Clusters can be interpreted as product groups with similar demand and sales behavior. "
                "For example, high-demand/high-variability items may require more safety stock."
            )
        else:
            st.warning("Need at least 2 features to compute 2D PCA.")
    else:
        st.warning("Click **Cluster Products** to run the inventory clustering analysis.")

# ---------------------------------------------------------------------
elif page == "About / Methods (Paper 2)":
    st.title("About This Project – Paper 2 View & Methods")

    st.markdown("""
    This part of the project is based on the literature review:

    **Darbanian, F., Brandtner, P., Falatouri, T., & Nasseri, M. (2024).  
    Data Analytics in Supply Chain Management: A State-of-the-Art Literature Review.  
    Operations and Supply Chain Management, 17(1).**

    ### Objectives

    - Map the concepts from the review article onto a **real dataset**  
    - Demonstrate how:
      - predictive analytics (RQ1, RQ3),
      - descriptive & clustering analytics (RQ2, RQ4),
      can be implemented in practice.

    ### Tools

    - Python, Pandas, NumPy  
    - Streamlit for the dashboard  
    - (Optionally) Scikit-learn for ML & clustering models  
    - Matplotlib / Altair for visualizations  

    ### Relationship to Paper 1

    - **Paper 1**: Method-focused deep learning framework (SOM + ANN + SHAP)  
    - **Paper 2**: Broad mapping of **where and how DA is used in SCM**  
    - This dashboard connects both:
      - Paper 1 informs model choices and interpretability  
      - Paper 2 informs problem selection (demand, logistics, inventory, segmentation)
    """)

    st.markdown("---")
    st.markdown("You can extend each RQ page with more advanced models and visualizations as needed for your assignment or thesis.")
