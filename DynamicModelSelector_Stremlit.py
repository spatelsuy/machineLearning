import streamlit as st
import numpy as np

# --- Symbolic models ---
def model_add(a, b): return a + b
def model_mul(a, b): return a * b
def model_pow_ab(a, b): return a ** b if a > 0 else float('inf')
def model_pow_ba(a, b): return b ** a if b > 0 else float('inf')
def model_sq_plus_b(a, b): return a * a + b
def model_a_plus_sq_b(a, b): return a + b * b

# --- Linear Regression ---
def linear_regression(train_data):
    X = [[a, b, 1] for a, b, _ in train_data]
    y = [res for _, _, res in train_data]
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    XTy = np.dot(XT, y)

    try:
        if np.linalg.det(XTX) != 0:
            w = np.linalg.solve(XTX, XTy)
        else:
            w, _, _, _ = np.linalg.lstsq(XTX, XTy, rcond=None)
        return lambda a, b: w[0]*a + w[1]*b + w[2], w
    except:
        return None, None

def evaluate_model(model_func, train_data):
    try:
        errors = [(model_func(a, b) - res) ** 2 for a, b, res in train_data]
        return sum(errors) / len(errors)
    except:
        return float('inf')

# --- Streamlit App ---
st.subheader("Dynamic Model Selector")
st.write("Train with 5 examples and find the best mathematical model!")

# --- Initialize session ---
if "train_data" not in st.session_state:
    st.session_state.train_data = []
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None

# --- Input training data ---
st.subheader("Enter Training Data (5 rows)")
with st.form("train_form"):
    inputs = []
    for i in range(5):
        col1, col2, col3 = st.columns(3)
        a = col1.number_input(f"a-{i+1}", min_value=0, max_value=100, step=1, format="%d", key=f"a_{i}")
        b = col2.number_input(f"b-{i+1}", min_value=0, max_value=100, step=1, format="%d", key=f"b_{i}")
        r = col3.number_input(f"result-{i+1}", min_value=0, max_value=100, step=1, format="%d", key=f"r_{i}")
        inputs.append((a, b, r))
    submitted = st.form_submit_button("Add Training Data")

if submitted:
    st.session_state.train_data.extend(inputs)
    st.success("Training data added!")

# --- Reset Data ---
if st.button("Reset All Training Data"):
    st.session_state.train_data = []
    st.session_state.best_model = None
    st.session_state.model_name = None
    st.experimental_rerun()

# --- Show Training Data ---
if st.session_state.train_data:
    st.subheader("Current Training Data")
    st.write(st.session_state.train_data)

    # --- Define models ---
    models = [
        ("Addition (a + b)", model_add),
        ("Multiplication (a * b)", model_mul),
        ("Power (a^b)", model_pow_ab),
        ("Power (b^a)", model_pow_ba),
        ("a^2 + b", model_sq_plus_b),
        ("a + b^2", model_a_plus_sq_b),
    ]

    lr_func, w = linear_regression(st.session_state.train_data)
    if lr_func:
        models.insert(0, ("Linear Regression", lr_func))

    # --- Evaluate models ---
    scored_models = [(name, func, evaluate_model(func, st.session_state.train_data)) for name, func in models]

    # --- Display scores ---
    st.subheader("Model Evaluation - (Mean Squared Error)")
    for name, _, mse in scored_models:
        st.write(f"**{name}** → MSE: {mse:.4f}")

    # --- Select best ---
    best = min(scored_models, key=lambda x: x[2])
    st.session_state.best_model = best[1]
    st.session_state.model_name = best[0]

    st.success(f"🏆 Best Model: **{best[0]}** with MSE = {best[2]:.4f} 🏆")

    if best[0] == "Linear Regression" and w is not None:
        st.write(f"**Learned Weights**: w0 = {w[0]:.2f}, w1 = {w[1]:.2f}, bias = {w[2]:.2f}")

# --- Prediction Section ---
if st.session_state.best_model:
    st.subheader("Make Prediction")
    col1, col2 = st.columns(2)
    a_test = col1.number_input("Enter a", key="predict_a")
    b_test = col2.number_input("Enter b", key="predict_b")

    if st.button("Predict"):
        pred = st.session_state.best_model(a_test, b_test)
        st.success(f"Prediction using **{st.session_state.model_name}**: {pred:.4f}")
