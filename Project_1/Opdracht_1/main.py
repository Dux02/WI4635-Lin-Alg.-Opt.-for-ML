from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set a random seed for reproducibility
RNG_SEED = 42

# a) 
def generate_data(samples: int, features: int, noise_std: float, seed = 79):
    rng = np.random.default_rng(seed)
    
    X = rng.uniform(0,1, size=(samples, features))
    true_weights = rng.uniform(0,1, size=features)
    noise = rng.normal(0, noise_std, size=samples)
    
    y = X @ true_weights + noise
    
    return X, y, true_weights

# b)
def train_test_split(X, y, test_size: float, seed = 79):
    rng = np.random.default_rng(seed)
    
    samples = X.shape[0]
    indices = rng.permutation(samples)
    
    test_samples = int(samples * test_size)
    
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


# c)
def fit_regression(X, y, lam: float):
    if lam < 0:
        raise ValueError("Regularization parameter lambda must be non-negative.")
    
    # Compute preprocessing statistics
    mean_X = np.mean(X, axis=0)
    x_scale = X.std(axis=0)
    x_scale = np.where(x_scale == 0, 1, x_scale)  # Avoid division by zero
    
    y_mean = y.mean()
    
    Z = (X - mean_X) / x_scale
    y_centered = y - y_mean
    
    if lam == 0:
        
        coefficients = np.linalg.lstsq(Z, y_centered, rcond=None)[0]
    else:
        # Ridge as an augmented LS problem
        p = Z.shape[1]
        
        Z_augmented = np.vstack([Z, np.sqrt(lam) * np.eye(p)])
        y_augmented = np.concatenate([y_centered, np.zeros(p)])
        
        coefficients = np.linalg.lstsq(Z_augmented, y_augmented, rcond=None)[0]
        
    return coefficients, mean_X, x_scale, y_mean

def fit_ordinary_least_squares(X, y):
    return fit_regression(X, y, lam=0)

def fit_ridge_regression(X, y, lam: float):
    return fit_regression(X, y, lam=lam)

def predict(model, X):
    coefficients, mean_X, x_scale, y_mean = model
    
    Z = (X - mean_X) / x_scale
    y_pred = Z @ coefficients + y_mean
    
    return y_pred

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def choose_lam(X, y, lam_values, n_folds: int, seed = 79):
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), n_folds)
    
    mean_errors = []
    
    for lam in lam_values:
        fold_errors = []
        
        for i in range(n_folds):
            test_indices = folds[i]
            train_indices = np.hstack([folds[j] for j in range(n_folds) if j != i])
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            
            model = fit_ridge_regression(X_train, y_train, lam)
            y_pred = predict(model, X_test)
            
            fold_errors.append(mean_squared_error(y_test, y_pred))
        
        mean_errors.append(np.mean(fold_errors))
    
    mean_errors = np.array(mean_errors)
    best_lam = lam_values[np.argmin(mean_errors)]
    
    return best_lam, mean_errors

# d)
def generate_collinear_data(X, samples: int, extra:int, features: int, noise_std: float, seed = 79):
    rng = np.random.default_rng(seed)
    
    combinations_weights = rng.uniform(-1, 1, size=(features,extra))
    
    collinear_features = (
        X @ combinations_weights + rng.normal(0, noise_std, size=(samples, extra))
    )
    
    X_collinear = np.column_stack([X, collinear_features])
    
    return X_collinear

# e)
def generate_irrelevant_features(X, samples: int, extra:int, seed = 79):
    rng = np.random.default_rng(seed)
    
    irrelevant_features = rng.uniform(0, 1, size=(samples, extra))
    
    X_irrelevant = np.column_stack([X, irrelevant_features])
    
    return X_irrelevant

def main():
    # a) Generate the original 300 × 20 data set.
    X, y, true_weights = generate_data(
        samples=300,
        features=20,
        noise_std=0.1,
        seed=RNG_SEED,
    )

    # d) Add 200 nearly collinear features.
    X_collinear = generate_collinear_data(
        X,
        samples=X.shape[0],
        extra=200,
        features=X.shape[1],
        noise_std=0.05,
        seed=RNG_SEED + 1,
    )

    # e) Add 200 irrelevant features to the original X.
    X_irrelevant = generate_irrelevant_features(
        X,
        samples=X.shape[0],
        extra=200,
        seed=RNG_SEED + 2,
    )

    # Candidate regularization strengths, including OLS (lambda = 0).
    lam_values = np.concatenate(([0.0], np.logspace(-4, 4, 41)))

    experiments = [
        ("Original data", X),
        ("Nearly collinear features", X_collinear),
        ("Irrelevant features", X_irrelevant),
    ]

    print(f"True weights:\n{true_weights}")

    for experiment_name, X_experiment in experiments:
        # b) Use the same row split for each experiment.
        X_train, X_test, y_train, y_test = train_test_split(
            X_experiment,
            y,
            test_size=0.2,
            seed=RNG_SEED,
        )

        # Select lambda using only the training set.
        best_lam, cv_errors = choose_lam(
            X_train,
            y_train,
            lam_values=lam_values,
            n_folds=5,
            seed=RNG_SEED + 3,
        )

        # c) Fit both models on the full training set.
        ols = fit_ordinary_least_squares(X_train, y_train)
        ridge = fit_ridge_regression(
            X_train, y_train, lam=best_lam
        )

        print(f"\n{experiment_name}")
        print(
            f"Features: {X_experiment.shape[1]} | "
            f"Train samples: {len(y_train)} | "
            f"Test samples: {len(y_test)}"
        )
        print(f"Selected lambda: {best_lam:.6g}")
        print(f"Best cross-validation MSE: {cv_errors.min():.6f}")

        for model_name, model in [("OLS", ols), ("Ridge", ridge)]:
            train_error = mean_squared_error(
                y_train, predict(model, X_train)
            )
            test_error = mean_squared_error(
                y_test, predict(model, X_test)
            )

            print(
                f"{model_name:5s} | "
                f"train MSE = {train_error:.6f} | "
                f"test MSE = {test_error:.6f}"
            )

def main_vis():
    output_dir = Path(__file__).resolve().parent
    image_dir = output_dir / "img"
    image_dir.mkdir(parents=True, exist_ok=True)
    X, y, true_weights = generate_data(
        samples=300,
        features=20,
        noise_std=0.1,
        seed=RNG_SEED,
    )

    X_collinear = generate_collinear_data(
        X,
        samples=X.shape[0],
        extra=200,
        features=X.shape[1],
        noise_std=0.05,
        seed=RNG_SEED + 1,
    )

    X_irrelevant = generate_irrelevant_features(
        X,
        samples=X.shape[0],
        extra=200,
        seed=RNG_SEED + 2,
    )

    lam_values = np.concatenate(([0.0], np.logspace(-4, 4, 41)))

    experiments = [
        ("Original data", X),
        ("Nearly collinear features", X_collinear),
        ("Irrelevant features", X_irrelevant),
    ]

    results = []
    cv_results = []

    for experiment_name, X_experiment in experiments:
        X_train, X_test, y_train, y_test = train_test_split(
            X_experiment,
            y,
            test_size=0.2,
            seed=RNG_SEED,
        )

        best_lam, cv_errors = choose_lam(
            X_train,
            y_train,
            lam_values=lam_values,
            n_folds=5,
            seed=RNG_SEED + 3,
        )

        models = [
            ("OLS", fit_ordinary_least_squares(X_train, y_train)),
            (
                "Ridge",
                fit_ridge_regression(X_train, y_train, best_lam),
            ),
        ]

        experiment_errors = []

        print(f"\n{experiment_name}")
        print(f"Selected lambda: {best_lam:.6g}")

        for model_name, model in models:
            train_error = mean_squared_error(
                y_train, predict(model, X_train)
            )
            test_error = mean_squared_error(
                y_test, predict(model, X_test)
            )

            experiment_errors.append([train_error, test_error])

            print(
                f"{model_name:5s} | "
                f"train MSE = {train_error:.6f} | "
                f"test MSE = {test_error:.6f}"
            )

        results.append(experiment_errors)
        cv_results.append((best_lam, cv_errors))

    # Shape: (experiments, models, train/test).
    results = np.array(results)

    # Plot 1: training and test errors.
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4), sharey=True,
        layout="constrained",
    )

    positions = np.arange(2)
    width = 0.35

    for i, (experiment_name, _) in enumerate(experiments):
        ax = axes[i]

        ax.bar(
            positions - width / 2,
            results[i, :, 0],
            width,
            label="Train",
            color="tab:blue",
        )
        ax.bar(
            positions + width / 2,
            results[i, :, 1],
            width,
            label="Test",
            color="tab:orange",
        )

        # The variance of the outcome noise is 0.1².
        ax.axhline(
            0.1 ** 2,
            color="black",
            linestyle="--",
            label="Noise variance",
        )

        ax.set_xticks(positions, ["OLS", "Ridge"])
        ax.set_title(experiment_name)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Mean squared error (log scale)")
    axes[0].legend()
    fig.suptitle("Training and test performance")
    fig.savefig(image_dir / "train_test_mse.png", dpi=200, bbox_inches="tight")

    # Plot 2: cross-validation error versus lambda.
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4),
        layout="constrained",
    )

    for i, (experiment_name, _) in enumerate(experiments):
        ax = axes[i]
        best_lam, cv_errors = cv_results[i]

        ax.plot(
            lam_values,
            cv_errors,
            marker=".",
            color="tab:blue",
        )

        ax.scatter(
            best_lam,
            cv_errors.min(),
            color="tab:red",
            zorder=3,
            label=f"Selected λ = {best_lam:.3g}",
        )

        # Unlike a logarithmic scale, symlog can display lambda = 0.
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_yscale("log")

        ax.set_title(experiment_name)
        ax.set_xlabel("λ")
        ax.set_ylabel("Cross-validation MSE")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("Choosing the ridge penalty")

    fig.savefig(image_dir / "cross_validation.png", dpi=200, bbox_inches="tight")
    print(f"Images saved to: {image_dir}")
    plt.show()

if __name__ == "__main__":
    main_vis()