"""
models.py
------------------------------------
Model training and evaluation for Lorenz System Prediction

Implements three predictive models of increasing complexity
    1. Linear Regression    -Linear Baseline
    2. Random Forest        -non linear classical ML
    3. LSTM                 -Sequence aware Deep Learning(Pytorch)
    
All these models are trained to predict the next state [x, y, z] from a window of past states. Evaluation measures how to 
prediction error grows with forecast horizon, testing the Lyapunov prediction ceiling.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------
#Reproducibility
# ---------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------

def prepare_sequences(
    trajectory: np.ndarray,
    n_lags: int = 50,
    test_fraction: float = 0.2,
) -> dict:
    """
    Prepare train/test splits from a Lorenz trajectory.
    
    Builds lag feature matrices and split chronologically, not randomly.
    Random splitting would leak future information into training, which is invalid for time series modeling.
    
    Args:
        trajectory: 2D array of shape (n_steps, 3) with columns [x, y, z]
        n_lags: Number of past timesteps to use as input features.
        test_fraction: Propotion of data reserved for testing. Default is 0.2
        
    Returns:
        Dictionary with keys:
            X_train, X_test: Features matrices of shape (n_samples, n_lags, 3)
            y_train, y_test: Target matrices of shape (n_samples, 3)
            scaler: Fitted StandardScaler (for inverse transforming predictions) 
    """
    #Build sequences: X[i] = trajectory[i:i+n_lags], y[i] = trajectory[i+n_lags]
    X, y = [], []
    for i in range(len(trajectory) - n_lags):
        X.append(trajectory[i:i + n_lags])
        y.append(trajectory[i + n_lags])
        
    X = np.array(X)
    y = np.array(y)
    
    # Chronological split
    split = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale using training statistics only, never fit on test data
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, 3)
    scaler.fit(X_train_2d)
    
    X_train = scaler.transform(X_train_2d.reshape(-1, 3)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, 3)).reshape(X_test.shape)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }
    
# ---------------------------------------------------------------------
# Model 1: Linear Regression
# ---------------------------------------------------------------------

def train_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> MultiOutputRegressor:

    """
    Train a multi output linear regression model

    Flattens the (n_samples, n_lags, 3) input into (n_samples, n_lags * 3) since sklearn expects 2D input.
    MultiOutput regressor fits one LinearRegression per output coordinate (x, y, z).
    
    Args:
        X_train: Trainig features of shape (n_samples, n_lags, 3)
        y_train: Training targets of shape (n_samples, 3).
        
    Returns:
        Fitted MultiOutputRegressor wrapping LinearRegression
    """
    X_flat = X_train.reshape(len(X_train), -1)
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_flat, y_train)
    return model

# ---------------------------------------------------------------------
# Model 2: Random Forest
# ---------------------------------------------------------------------  

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
) -> MultiOutputRegressor:
    
    """
    Train a Multioputput Random Forest Regressor.
    
    Random Forest captures nonlinear relationships betwenn past states and future states without any asumptions about
    the functional form. Each tree votes independently and their average forms the prediction.
    
    Args:
        X_train: Training features of shape (n_samples, n_lags, 3)
        y_train: Training targets of shape (n_samples, 3)
        n_estimators: Number of trees in the forest. Default is 100.
        
    Returns: 
        Fitted MultiOutputRegressor wrapping RandomForestRegressor.
    """
    X_flat = X_train.reshape(len(X_train), -1)
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators= 100,
            random_state= RANDOM_SEED,
            n_jobs= -1
        )
    )
    
    model.fit(X_flat, y_train)
    return model  

# ---------------------------------------------------------------------
# Model 3: LSTM (PyTorch)
# --------------------------------------------------------------------- 

class LorenzLSTM(nn.Module):
    """
    LSTM network for Lorenz State Prediction.
    
    Takes a sequence of past states and predicts the next state.
    Architecture: LSTM -> Dropout -> Linear Output Layer.
    
    The LSTM processes the input sequence step by step, maintining a hidden state that summarizes 
    temporal context, making it naturally suited for chaotic time series.
    
    Args:
        input_size: Number of features per timestep. Default is 3 (x, y, z).
        hidden_size: Number of LSTM hidden units. Default is 64.
        num_layers: Number of stacked LSTM layers. Default is 2
        output_size: Number of predicted coordinates. Default is 3.
        dropout: Dropout rate between LSTM layers. Default is 0.2
    """
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super(LorenzLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size= input_size,
            hidden_size= hidden_size,
            num_layers= num_layers,
            batch_first= True,
            dropout= dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM and linear layer.
        
        Args:
            x: Input tensor of shape (batch_size, n_lags, input_size)
            
        Returns: 
            Output tensor of shape (batch_size, output_size)        
        """
        lstm_out, _ = self.lstm(x)
        # Take only the last timestep's output
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)
        
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3
) -> tuple[LorenzLSTM, list[float]]:
    
    """
     Train the LSTM model on Lorenz prediction data.

    Uses Adam optimizer and MSE loss. Training loss is tracked per epoch
    for plotting the learning curve.

    Args:
        X_train: Training features of shape (n_samples, n_lags, 3).
        y_train: Training targets of shape (n_samples, 3).
        hidden_size: LSTM hidden units. Default is 64.
        num_layers: Stacked LSTM layers. Default is 2.
        epochs: Training epochs. Default is 30.
        batch_size: Samples per gradient update. Default is 256.
        learning_rate: Adam learning rate. Default is 1e-3.

    Returns:
        model: Trained LorenzLSTM instance.
        losses: List of mean training loss per epoch.
    """
    
    # Convert to PyTorch Tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size= batch_size, shuffle= True)
    
    model = LorenzLSTM (hidden_size= hidden_size, num_layers= num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} — Loss: {mean_loss:.6f}")

    return model, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "sklearn",
) -> float:
    """
    Compute test MSE for a trained model.

    Args:
        model: Trained model (sklearn or LorenzLSTM).
        X_test: Test features of shape (n_samples, n_lags, 3).
        y_test: Test targets of shape (n_samples, 3).
        model_type: Either 'sklearn' or 'lstm'. Default is 'sklearn'.

    Returns:
        Mean squared error on the test set.
    """
    if model_type == "sklearn":
        X_flat = X_test.reshape(len(X_test), -1)
        y_pred = model.predict(X_flat)
    else:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            y_pred = model(X_tensor).numpy()

    return float(mean_squared_error(y_test, y_pred))


def evaluation_multistep(
    model,
    trajectory: np.ndarray,
    scaler: StandardScaler,
    n_lags: int = 50,
    max_steps: int = 500,
    n_trials: int = 20,
    model_type: str = "sklearn",
) -> tuple[np.ndarray, np.ndarray]:

    """
    Evaluate prediction error as a function of forecast horizon.

    For each trial, picks a random starting point, feeds the model its own autoregressively and measures how error
    grows with forecast horizon. This reveals the Lyapunov ceiling.

    Args:
        model: Trained model (sklearn or LorenzLSTM).
        trajectory: Full unscaled trajectory array (n_steps, 3).
        scaler: Fitted StandardScaler from prepare_sequences().
        n_lags: Input window size used during training.
        max_steps: Maximum forecast horizon to evaluate.
        n_trials: Number of random starting points to average over.
        model_type: Either 'sklearn' or 'lstm'.
        
    Returns:
        horizons: Array of step indices [1, 2, ..., max_steps].
        mean_errors: Mean MSE at each horizon, averaged over trials.
    """

    np.random.seed(RANDOM_SEED)
    n = len(trajectory)
    all_errors = []

    for _ in range(n_trials):
        # Pick a random starting point with enough room ahead
        start = np.random.randint(n_lags, n - max_steps - 1)
        
        # Seed window-scaled
        window = scaler.transform(trajectory[start:start + n_lags])
        true_future = scaler.transform(trajectory[start + n_lags:start + n_lags + max_steps])
        
        predictions = []
        current_window = window.copy()
        
        for step in range(max_steps):
                if model_type == "sklearn":
                    X_input = current_window.reshape(1, -1)
                    next_pred = model.predict(X_input)[0]
                else:
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(current_window).unsqueeze(0)
                        next_pred = model(X_tensor).numpy()[0]

                predictions.append(next_pred)
                # Slide window forward — drop oldest, append prediction
                current_window = np.vstack([current_window[1:], next_pred])

        predictions = np.array(predictions)
        errors = np.mean((predictions - true_future) ** 2, axis=1)
        all_errors.append(errors)

    horizons = np.arange(1, max_steps + 1)
    mean_errors = np.mean(all_errors, axis=0)
    return horizons, mean_errors
        


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from simulate import simulate_lorenz
    import matplotlib.pyplot as plt

    print("Simulating Lorenz system...")
    t, trajectory = simulate_lorenz(t_end=100.0, n_steps=50000)
    dt = t[1] - t[0]

    print("Preparing sequences...")
    data = prepare_sequences(trajectory, n_lags=50)
    print(f"  Train: {data['X_train'].shape}, Test: {data['X_test'].shape}")

    print("\nTraining Linear Regression...")
    lr_model = train_linear_model(data["X_train"], data["y_train"])
    lr_mse   = evaluate_model(lr_model, data["X_test"], data["y_test"])
    print(f"  One-step MSE: {lr_mse:.6f}")

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(data["X_train"], data["y_train"])
    rf_mse   = evaluate_model(rf_model, data["X_test"], data["y_test"])
    print(f"  One-step MSE: {rf_mse:.6f}")

    print("\nTraining LSTM...")
    lstm_model, losses = train_lstm(data["X_train"], data["y_train"], epochs=30)
    lstm_mse = evaluate_model(lstm_model, data["X_test"], data["y_test"],
                              model_type="lstm")
    print(f"  One-step MSE: {lstm_mse:.6f}")

    print("\nEvaluating multi-step prediction horizon...")
    horizons, lr_errors   = evaluation_multistep(
        lr_model, trajectory, data["scaler"], max_steps= 700, model_type="sklearn")
    _, rf_errors          = evaluation_multistep(
        rf_model, trajectory, data["scaler"], max_steps= 700,model_type="sklearn")
    _, lstm_errors        = evaluation_multistep(
        lstm_model, trajectory, data["scaler"], max_steps= 700, model_type="lstm")

    # Convert steps to time units
    time_horizons = horizons * dt

    plt.figure(figsize=(12, 5))
    plt.semilogy(time_horizons, lr_errors,   label="Linear Regression", color="steelblue")
    plt.semilogy(time_horizons, rf_errors,   label="Random Forest",     color="tomato")
    plt.semilogy(time_horizons, lstm_errors, label="LSTM",              color="seagreen")
    plt.axvline(x=1/0.8746, color="black", linestyle="--", linewidth=1.2,
                label=f"Lyapunov horizon ≈ {1/0.8746:.2f} time units")
    plt.title("Prediction Error vs Forecast Horizon", fontsize=13)
    plt.xlabel("Forecast horizon (time units)")
    plt.ylabel("Mean Squared Error (log scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/prediction_horizon.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved to reports/figures/prediction_horizon.png")