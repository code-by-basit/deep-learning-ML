import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import joblib

data = {
    "feature1": [1.2, 2.0, 3.1, 4.5, 0.5, 1.8, 2.9, 3.6, 5.1, 2.3],
    "feature2": [3.4, 1.0, 0.9, 2.1, 4.0, 2.2, 1.5, 0.8, 1.1, 3.0],
    "feature3": [2.1, 0.5, 4.2, 3.0, 1.2, 2.8, 3.1, 4.9, 0.2, 1.7],
    "target": [10.5, 5.2, 12.3, 15.1, 6.0, 9.8, 11.4, 14.2, 11.0, 8.9]
}

df = pd.DataFrame(data)

X = df[["feature1", "feature2", "feature3"]].values
y = df["target"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalar = StandardScaler()
X_train_scaler = scalar.fit_transform(X_train)
X_test_scaler = scalar.transform(X_test)

X_train = torch.tensor(X_train_scaler, dtype=torch.float32)
X_test = torch.tensor(X_test_scaler, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# neural network Architecture
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # self.network = nn.Sequential(
        #     nn.Linear(3, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )
        #  3 - 32 - 16 - 1
        self.input_layer = nn.Linear(3, 32)
        self.relu1 == nn.ReLU()
        self.hidden_layer1 = nn.Linear(32, 16)
        self.relu2 == nn.ReLU()
        self.output_layer == nn.Linear(16, 1)
    
    def forward(self, x):
        # return self.network(x)
        x = self.input_layer(x)
        x = self.relu1(x)
        x = self.hidden_layer1(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    print("--- Training Started ---")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            y_prediction = model(batch_X)
            loss = criterion(y_prediction, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_X.size(0)

    print("--- Training Complete ---")
        
    torch.save(model.state_dict(), "mlp_model.pth")
    joblib.dump(scalar, "scaler.joblib")
    print("Model weights successfully saved to mlp_model.pth!")

    # print("\n--- Running Inference ---")
    # new_sample = [[2.5, 1.8, 3.0]]

    # new_sample_scaled = scalar.transform(new_sample)
    # new_tensor = torch.tensor(new_sample_scaled, dtype=torch.float32).to(device)

    # model.eval()
    # with torch.no_grad():
    #     raw_prediction = model(new_tensor)
        
    #     target = torch.tensor([[12.2]], dtype=torch.float32).to(device)
    #     inference_loss = criterion(raw_prediction, target)

    #     final_prediction = raw_prediction.cpu().item()
    #     final_loss = inference_loss.cpu().item()

    #     print(f"Input features:           {new_sample[0]}")
    #     print(f"Predicted Target Value:   {final_prediction:.4f}")
    #     print(f"Evaluation Target Ground Truth: 12.2000")
    #     print(f"Inference MSE Loss:       {final_loss:.4f}")


