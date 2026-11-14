
import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="QuantumGuard AI",
    page_icon="⚛️",
    layout="wide"
)

# =====================================================
# EXPERIMENT RESULTS - LATEST RETRAINED RESULTS
# =====================================================

results = pd.DataFrame([
    ["Breast Cancer", "Classical OCSVM", 0.9230, 0.9181, 0.8837, 0.3662, None],
    ["Breast Cancer", "Quantum Hybrid", 0.7216, 0.7240, 0.4419, 0.0563, -0.003463],
    ["Breast Cancer", "Quantum QAE", 0.8097, 0.7830, 0.7674, 0.3380, 0.151492],
    ["Wine", "Classical OCSVM", 0.9931, 0.9968, 0.9583, 0.0833, None],
    ["Wine", "Quantum Hybrid", 0.8403, 0.9282, 0.6667, 0.0000, -0.006660],
    ["Wine", "Quantum QAE", 0.7604, 0.9009, 0.6667, 0.0833, 0.654812]
], columns=[
    "Dataset",
    "Model",
    "AUROC",
    "AUPRC",
    "Recall",
    "FPR",
    "Threshold"
])

# =====================================================
# QUANTUM TRAINING COSTS
# =====================================================

training_costs = pd.DataFrame({
    "Dataset": ["Breast Cancer", "Wine"],
    "Training Time (seconds)": [75.7744, 12.6133],
    "Objective Evaluations": [150, 40]
})

# =====================================================
# QUANTUM CIRCUIT RESOURCES
# =====================================================

resource_data = pd.DataFrame({
    "Gate": ["RY", "RZ", "CNOT"],
    "Count": [12, 8, 6]
})

# =====================================================
# TRAINING LOSS
# =====================================================

epochs = list(range(1, 11))

breast_loss = [
    0.391337, 0.233033, 0.177745, 0.164603, 0.160084,
    0.157800, 0.157037, 0.156675, 0.156544, 0.156543
]

wine_loss = [
    0.707587, 0.688761, 0.670036, 0.650242, 0.630148,
    0.607229, 0.580560, 0.550245, 0.515834, 0.478594
]

loss_df = pd.DataFrame({
    "Epoch": epochs,
    "Breast Cancer": breast_loss,
    "Wine": wine_loss
})

# =====================================================
# INFERENCE TIME - EARLIER MEASUREMENTS
# =====================================================

timing = pd.DataFrame({
    "Dataset": ["Breast Cancer", "Wine"],
    "Average Time (ms)": [34.10, 15.40]
})

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("⚛️ QuantumGuard AI")

st.sidebar.caption(
    "Quantum-Classical Anomaly Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Comparison",
        "Quantum Analysis",
        "Research Results"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Experimental Prototype\n\n"
    "PennyLane + Scikit-learn + Streamlit"
)

# =====================================================
# MAIN HEADER
# =====================================================

st.title("⚛️ QuantumGuard AI")

st.subheader(
    "Quantum-Classical Anomaly Detection Platform"
)

st.markdown("""
An experimental research platform comparing classical
One-Class SVM, Quantum Autoencoder (QAE), and
Hybrid Quantum-Classical anomaly detection models.
""")

st.divider()

# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "Overview":

    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Datasets", "2")
    c2.metric("Models", "3")
    c3.metric("Quantum Qubits", "4")
    c4.metric("Latent Qubits", "2")

    st.divider()

    st.subheader("Experimental Datasets")

    d1, d2 = st.columns(2)

    with d1:

        st.markdown("### Breast Cancer Wisconsin")

        st.write("**Total Samples:** 569")
        st.write("**Original Features:** 30")
        st.write("**Inliers:** Benign samples")
        st.write("**Anomalies:** Malignant samples")

    with d2:

        st.markdown("### Wine Dataset")

        st.write("**Total Samples:** 178")
        st.write("**Original Features:** 13")
        st.write("**Inliers:** Class 0")
        st.write("**Anomalies:** Classes 1 and 2")

    st.divider()

    st.subheader("Model Performance")

    fig = px.bar(
        results,
        x="Model",
        y="AUROC",
        color="Dataset",
        barmode="group",
        title="AUROC Comparison Across Datasets",
        range_y=[0, 1],
        text_auto=".3f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Performance Summary")

    best = results.loc[
        results.groupby("Dataset")["AUROC"].idxmax()
    ]

    st.dataframe(
        best[["Dataset", "Model", "AUROC", "AUPRC"]],
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# PAGE 2: MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.header("Model Performance Comparison")

    dataset = st.selectbox(
        "Select Dataset",
        ["All Datasets", "Breast Cancer", "Wine"]
    )

    filtered = results.copy()

    if dataset != "All Datasets":

        filtered = filtered[
            filtered["Dataset"] == dataset
        ]

    st.subheader("Performance Metrics")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    metric = st.selectbox(
        "Select Performance Metric",
        ["AUROC", "AUPRC", "Recall", "FPR"]
    )

    fig = px.bar(
        filtered,
        x="Model",
        y=metric,
        color="Model",
        title=f"{metric} Comparison",
        range_y=[0, 1],
        text_auto=".3f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Model Thresholds")

    st.dataframe(
        filtered[["Dataset", "Model", "Threshold"]],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Results CSV",
        data=results.to_csv(index=False),
        file_name="quantumguard_results.csv",
        mime="text/csv"
    )

# =====================================================
# PAGE 3: QUANTUM ANALYSIS
# =====================================================

elif page == "Quantum Analysis":

    st.header("Quantum Circuit Analysis")

    st.subheader("Quantum Autoencoder Architecture")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Qubits", "4")
    c2.metric("Latent Qubits", "2")
    c3.metric("Trash Qubits", "2")

    st.markdown("""
    **Architecture:**

    - Four-qubit quantum circuit.
    - Two latent qubits retain compressed quantum information.
    - Two trash qubits form the discarded subsystem.
    - Trainable RY and RZ rotations.
    - CNOT gates provide entanglement.
    """)

    st.divider()

    # Circuit resources

    st.subheader("Circuit Resources")

    fig = px.bar(
        resource_data,
        x="Gate",
        y="Count",
        color="Gate",
        title="Quantum Gate Distribution",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    r1, r2, r3 = st.columns(3)

    r1.metric("Circuit Depth", "10")
    r2.metric("Total Gates", "26")
    r3.metric("Qubits", "4")

    st.write("**Simulation Backend:** PennyLane default.qubit")
    st.write("**Measurement:** Analytic expectation/probability")
    st.write("**Finite Shots:** Not used")

    st.warning("""
    These are quantum simulator results, not measurements
    from real quantum hardware.
    """)

    st.divider()

    # Training loss

    st.subheader("Quantum Autoencoder Training Loss")

    fig = px.line(
        loss_df,
        x="Epoch",
        y=["Breast Cancer", "Wine"],
        markers=True,
        title="Training Loss Across Epochs"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption("""
    Loss represents the probability of the trash subsystem
    not being in the |00⟩ state. It is not classical input
    reconstruction MSE.
    """)

    st.divider()

    # Training cost

    st.subheader("Quantum Training Cost")

    st.dataframe(
        training_costs,
        use_container_width=True,
        hide_index=True
    )

    st.caption("""
    Objective evaluations represent calls to the training
    objective function, not the exact number of underlying
    quantum circuit executions.
    """)

    st.divider()

    # Inference timing

    st.subheader("Quantum Inference Time")

    st.dataframe(
        timing,
        use_container_width=True,
        hide_index=True
    )

    st.caption("""
    These are earlier Colab simulator measurements.
    They should be remeasured using the latest trained weights.
    They are not quantum hardware timings.
    """)

# =====================================================
# PAGE 4: RESEARCH RESULTS
# =====================================================

elif page == "Research Results":

    st.header("Research Findings")

    st.subheader("Key Observations")

    st.markdown("""
    - Classical One-Class SVM achieved the highest AUROC
      and AUPRC on both datasets.
    - Quantum Hybrid achieved a lower false-positive rate
      than the classical model on both datasets.
    - Quantum Hybrid achieved zero false positives on
      the Wine test set.
    - Quantum QAE demonstrated a trash-subsystem
      compression objective using latent and trash qubits.
    - Quantum models showed different anomaly-detection
      characteristics across datasets.
    - No quantum advantage has been demonstrated in
      these experiments.
    """)

    st.divider()

    st.subheader("Best AUROC by Dataset")

    best = results.loc[
        results.groupby("Dataset")["AUROC"].idxmax()
    ]

    st.dataframe(
        best[["Dataset", "Model", "AUROC", "AUPRC"]],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Research Limitations")

    st.markdown("""
    1. Experiments were performed using the PennyLane simulator.
    2. No quantum hardware advantage has been demonstrated.
    3. The quantum autoencoder uses a trash-qubit compression
       objective, not decoder-based classical reconstruction.
    4. Results depend on the selected train, validation and test split.
    5. Objective function evaluations were recorded, but exact
       underlying quantum circuit execution counts were not measured.
    6. Inference timings need to be remeasured after the latest
       QAE retraining.
    """)

    st.divider()

    st.success(
        "QuantumGuard AI experimental results dashboard is ready."
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "QuantumGuard AI | Experimental Research Project | "
    "PennyLane + Python + Scikit-learn + Streamlit"
)
