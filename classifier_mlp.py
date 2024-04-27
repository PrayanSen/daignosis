import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from read_neo4j import neo_driver, extract_features


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_model(model, criterion, optimizer, train_loader, num_epochs, device):
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()


def predict(model, test_loader):
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs)
            predictions.extend(predicted.numpy())
    return predictions


if __name__ == "__main__":
    driver = neo_driver()
    query = """
            MATCH (sample:Biological_sample)-[:HAS_DISEASE]->(disease:Disease)
            OPTIONAL MATCH (sample)-[:HAS_DAMAGE]->(damage:Gene)
            OPTIONAL MATCH (sample)-[:HAS_PHENOTYPE]->(phenotype:Phenotype)
            RETURN sample, disease, damage, phenotype
        """
    df = extract_features(driver, query)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: ", device)

    data = df.drop(columns=["disease_name", "disease_synonyms"])
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
    data = pd.concat([data.drop(columns=categorical_cols), encoded_data], axis=1)

    # Split data into features and labels
    labels = data[["subject_id", "class"]]
    X = data.drop(columns=["class"])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train["class"].values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_size = X_train.shape[1]
    model = MLP(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, criterion, optimizer, train_loader, num_epochs=10, device=device)

    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions = predict(model, test_loader)

    f1 = f1_score(y_test["class"], predictions)

    res = {
        "subject_id": y_test["subject_id"],
        "disease": predictions
    }
    res_df = pd.DataFrame(res)
    res_df.to_csv("output_task_a.csv", index=False)
