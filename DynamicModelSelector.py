# dynamic_model_selector.py
# This program identifies the best-fit mathematical model from training data.

import math

# Symbolic models
def model_add(a, b): return a + b
def model_mul(a, b): return a * b
def model_pow_ab(a, b): return a ** b if a > 0 else float('inf')
def model_pow_ba(a, b): return b ** a if b > 0 else float('inf')
def model_sq_plus_b(a, b): return a * a + b
def model_a_plus_sq_b(a, b): return a + b * b



def power_regression(train_data):
    import numpy as np
    import math

    # Filter out invalid data (a, b, result must be > 0 to take log)
    filtered_data = [(a, b, r) for a, b, r in train_data if a > 0 and b > 0 and r > 0]
    if not filtered_data or len(filtered_data) < 3:
        print("Not enough valid data points (a, b, result must be > 0).")
        return None

    X = [[math.log(a), math.log(b), 1] for a, b, _ in filtered_data]
    y = [math.log(r) for _, _, r in filtered_data]

    XT = list(zip(*X))
    XTX = [[sum(xi * xj for xi, xj in zip(row, col)) for col in zip(*X)] for row in XT]
    XTy = [sum(x * y[i] for i, x in enumerate(col)) for col in XT]

    try:
        if np.linalg.det(np.array(XTX)) != 0:
            w = np.linalg.solve(np.array(XTX), np.array(XTy))
        else:
            w, _, _, _ = np.linalg.lstsq(np.array(XTX), np.array(XTy), rcond=None)

        # Return the power model: result = exp(w2) * a^w0 * b^w1
        return lambda a, b: math.exp(w[2]) * (a ** w[0]) * (b ** w[1])
    except Exception as e:
        print("Power regression failed:", e)
        return None


def exponential_regression(train_data):
    import numpy as np
    
    # Filter out invalid data (result must be > 0 to take log)
    filtered_data = [(a, b, r) for a, b, r in train_data if r > 0]
    if not filtered_data or len(filtered_data) < 3:
        print("Not enough valid data points (result must be > 0).")
        return None

    X = [[a, b, 1] for a, b, _ in filtered_data]
    y = [math.log(r) for _, _, r in filtered_data]

    XT = list(zip(*X))
    XTX = [[sum(xi * xj for xi, xj in zip(row, col)) for col in zip(*X)] for row in XT]
    XTy = [sum(x * y[i] for i, x in enumerate(col)) for col in XT]

    try:
        if np.linalg.det(np.array(XTX)) != 0:
            w = np.linalg.solve(np.array(XTX), np.array(XTy))
        else:
            w, _, _, _ = np.linalg.lstsq(np.array(XTX), np.array(XTy), rcond=None)
        
        # Return the exponential model: result = exp(w0 * a + w1 * b + w2)
        return lambda a, b: math.exp(w[0] * a + w[1] * b + w[2])
    except Exception as e:
        print("Exponential regression failed:", e)
        return None



# Linear Regression Model
def linear_regression(train_data):
    n = len(train_data)
    X = [[a, b, 1] for a, b, _ in train_data]
    y = [res for _, _, res in train_data]

    XT = list(zip(*X))
    XTX = [[sum(xi * xj for xi, xj in zip(row, col)) for col in zip(*X)] for row in XT]
    XTy = [sum(x * y[i] for i, x in enumerate(col)) for col in XT]

    # Solve linear system (XTX * w = XTy)
    try:
        import numpy as np
        # w = np.linalg.solve(np.array(XTX), np.array(XTy))
        if np.linalg.det(np.array(XTX)) != 0:
            w = np.linalg.solve(np.array(XTX), np.array(XTy))
            print("W=\n", w)
        else:
            print("Matrix is singular; using least squares solution instead.")
            w, _, _, _ = np.linalg.lstsq(np.array(XTX), np.array(XTy), rcond=None)
            print("W in else=\n", w)
            
        return lambda a, b: w[0] * a + w[1] * b + w[2]
    except:
        return None

# Evaluate model with Mean Squared Error
def evaluate_model(model_func, train_data):
    try:
        errors = [(model_func(a, b) - res) ** 2 for a, b, res in train_data]
        return sum(errors) / len(errors)
    except:
        return float('inf')

def mainMethod():
    print("Enter 5 training examples as a,b,result (e.g. 2,3,8)")
    train_data = []
    while len(train_data) < 5:
        try:
            entry = input(f"Example {len(train_data)+1}: ")
            a, b, r = map(float, entry.strip().split(","))
            train_data.append((a, b, r))
        except:
            print("Invalid input. Please enter in format a,b,result")

    # Symbolic models
    models = [
        ("Addition (a + b)", model_add),
        ("Multiplication (a * b)", model_mul),
        ("Power (a^b)", model_pow_ab),
        ("Power (b^a)", model_pow_ba),
        ("a^2 + b", model_sq_plus_b),
        ("a + b^2", model_a_plus_sq_b),
    ]

    # Add learnable model
    lr_model = linear_regression(train_data)
    if lr_model:
        models.insert(0, ("Linear Regression", lr_model))
        
    ex_model = exponential_regression(train_data)
    if ex_model:
        models.insert(1, ("Exponential Regression", ex_model))
        
    po_model = power_regression(train_data)
    if po_model:
        models.insert(2, ("Power Regression", po_model))

    # Evaluate and select best
    scored_models = [(name, model, evaluate_model(model, train_data)) for name, model in models]
    for name, _, error in scored_models:
        print(f"Model: {name}, MSE (Mean Squared Error): {error}")
    
    best_model = min(scored_models, key=lambda x: x[2])

    print(f"\nBest model: {best_model[0]}")
    print(f"MSE (Mean Squared Error): {best_model[2]:.4f}")

    # Predict loop
    while True:
        test = input("\nEnter a,b to predict (or 'exit'): ")
        if test.lower() == "exit":
            break
        try:
            a, b = map(float, test.strip().split(","))
            print(f"Prediction: {best_model[1](a, b)}")
        except:
            print("Invalid input. Use format: a,b")

if __name__ == "__main__":
    mainMethod()
