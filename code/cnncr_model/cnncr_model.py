"""CNN-CR model definition and helpers, extracted verbatim from CNN_CR_model_train_pytorch.py
(the script used for the paper). Hyperparameters are the paper values. NUM_BRANCHES = 20 for the
promoter + enhancer model and 10 for the promoter-only model (pass num_branches explicitly)."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss

NUM_MOTIFS = 256
MOTIF_LENGTH = 8
MAXPOOL0_SIZE = 5
MAXPOOL1_SIZE = 2
MAXPOOL2_SIZE = 3
MAXPOOL3_SIZE = 4
CONV1_FILTER = 64
CONV2_FILTER = 64
CONV3_FILTER = 64
REG1 = 0.1
REG2 = 0.2
REG3 = 0.3
DROPOUT1 = 0.15
DROPOUT2 = 0.15
# training hyperparameters, exactly as in CNN_CR_model_train_pytorch.py
batch = 64
epoch = 100
lr = 0.0001
decay = 0
patience = 10
NUM_BRANCHES = 20

def cross_val_gene(data, y, iteration):
    # Reshape data and y to shape (18356, x, 20, 400, 4)
    reshaped_data = np.transpose(data, (1, 0, 2, 3, 4))
    reshaped_y = np.transpose(y)

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(reshaped_data, reshaped_y, test_size=0.1, random_state=42)

    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Retrieve training and validation sets for the specified fold
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        if fold == iteration:
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            break  # Stop after retrieving the specified fold

    return X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test, y_test

# Custome bias layer
class BiasLayer(nn.Module):
    def __init__(self, num_features):
        super(BiasLayer, self).__init__()
        # self.bias = Parameter(torch.zeros(num_features))  # Bias parameter
        self.weight = Parameter(torch.ones(num_features))  # Weight parameter

    def forward(self, x):
        return x * self.weight
        # return x + self.bias
        # return (x + self.bias) * self.weight

class MultiBranchModel(nn.Module):
    def __init__(self, num_branches=NUM_BRANCHES):
        super(MultiBranchModel, self).__init__()
        # Assuming each branch takes input of shape (400, 4)
        self.branches = nn.ModuleList([CNNBranch(branch_number=i) for i in range(num_branches)])

    def forward(self, x):
        # Convert x to float, if not already
        x = x.float()
        # x is expected to have shape [batch_size, 20, 400, 4]
        batch_size = x.size(0)
        
        # Initialize outputs and mask for non-zero input tracking
        branch_outputs = torch.zeros(batch_size, len(self.branches), device=x.device)
        contextual_outputs = []
        combined_outputs = []

        for i, branch in enumerate(self.branches):
            branch_input = x[:, i, :, :]
            branch_output, contextual_output, combined_output = branch(branch_input)
            branch_outputs[:, i] = branch_output.squeeze()
            contextual_outputs.append(contextual_output.squeeze())
            combined_outputs.append(combined_output.squeeze())

        final_output = branch_outputs.sum(dim=1, keepdim=True)

        return final_output, torch.stack(contextual_outputs, dim=1), torch.stack(combined_outputs, dim=1)
    
class CNNBranch(nn.Module):
    def __init__(self, branch_number):
        super(CNNBranch, self).__init__()
        self.branch_number = branch_number
        
        # Initial Convolution
        self.conv0 = nn.Conv1d(in_channels=4, out_channels=NUM_MOTIFS, kernel_size=MOTIF_LENGTH, padding='valid', bias=False)   
        init.xavier_normal_(self.conv0.weight)
        # init.kaiming_normal_(self.conv0.weight, nonlinearity='relu')


        # Continuing the main path
        self.maxpool0 = nn.MaxPool1d(MAXPOOL0_SIZE)
        
        self.conv1 = nn.Conv1d(in_channels=NUM_MOTIFS, out_channels=CONV1_FILTER, kernel_size=8, padding='same')
        init.xavier_normal_(self.conv1.weight)
        # init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.maxpool1 = nn.MaxPool1d(MAXPOOL1_SIZE)
        
        self.conv2 = nn.Conv1d(in_channels=CONV1_FILTER, out_channels=CONV2_FILTER, kernel_size=8, padding='same')
        init.xavier_normal_(self.conv2.weight)
        # init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.maxpool2 = nn.MaxPool1d(MAXPOOL2_SIZE)
        
        self.conv3 = nn.Conv1d(in_channels=CONV2_FILTER, out_channels=CONV3_FILTER, kernel_size=8, padding='same')
        init.xavier_normal_(self.conv3.weight)
        # init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        self.maxpool3 = nn.MaxPool1d(MAXPOOL3_SIZE)
        
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(192, NUM_MOTIFS) 
        self.dropout1 = nn.Dropout(DROPOUT1)
        
        self.dense2 = nn.Linear(NUM_MOTIFS, NUM_MOTIFS)
        self.dropout2 = nn.Dropout(DROPOUT2)

        self.contextual_weight = nn.Linear(NUM_MOTIFS, NUM_MOTIFS)
        self.bias_layer = BiasLayer(1)
    
    def forward(self, x):

        # is_non_zero_sequence = torch.any(x.view(x.size(0), -1) != 0, dim=1).unsqueeze(-1).float()
        is_non_zero_sequence = torch.any(x[:, 0, :] != 0, dim=1).unsqueeze(-1).float()

        x = x.permute(0, 2, 1)
        # Initial motif detection
        x = self.conv0(x)
        # x = torch.exp(x)
        # x = F.relu(x)

        # Get mean motif activation
        mean_activation = torch.mean(x, dim=2)
                
        # Main path
        # x = torch.exp(x)
        x = F.relu(x)
        x = self.maxpool0(x)
        
        x = self.conv1(x)
        # x = F.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        # x = F.relu(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        # x = F.relu(x)
        x = self.maxpool3(x)
        
        x = self.flatten(x)

        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Rejoin paths
        contextual_weight = self.contextual_weight(x)
        contextual_weight *= is_non_zero_sequence  # This zeroes out contextual weights for all-zero sequences

        combined = contextual_weight * mean_activation
        combined_sum = combined.sum(dim=1, keepdim=True)  # Sum across motifs
        output = self.bias_layer(combined_sum)

        return output, contextual_weight, combined

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.patience_counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.patience:
            return True
        return False

def l2_regularization_loss(parameters, strength):
    l2_loss = 0.0
    for param in parameters:
        l2_loss += torch.sum(param ** 2)
    return strength * l2_loss

def l1_regularization_loss(parameters, strength):
    l1_loss = 0.0
    for param in parameters:
        l1_loss += torch.sum(torch.abs(param))  # Use torch.abs for L1 loss
    return strength * l1_loss

def l2_activity_loss(output, strength):
    # Calculate L2 loss on the activity (output) of a layer
    return torch.sum(output ** 2) * strength

def l1_activity_loss(output, strength):
    # Calculate L1 loss on the activity (output) of a layer
    return torch.sum(torch.abs(output)) * strength

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device):
    early_stopping = EarlyStopping(patience=patience)
    # Lists to keep track of losses
    train_losses = []
    val_losses = []
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0  # Accumulator for the total training loss

        for X_batch, Y_batch in train_dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(X_batch)
            loss = criterion(outputs, Y_batch)

            l2_loss = 0
            for branch in model.branches:
                # Accumulate L2 loss for each set of parameters and their respective regularization coefficient
                l2_loss += l2_regularization_loss(branch.dense1.parameters(), REG1)
                l2_loss += l2_regularization_loss(branch.dense2.parameters(), REG2)
                l2_loss += l2_regularization_loss(branch.contextual_weight.parameters(), REG3)

            # Activity regularization loss for the contextual_weight layer of each branch
            # activity_loss = sum(l2_activity_loss(co, REG3) for co in contextual_outputs)

            # total_batch_loss = loss + l2_loss + activity_loss
            total_batch_loss = loss + l2_loss

            total_batch_loss.backward()
            optimizer.step()

            # Apply non-negativity constraint by clamping the weights
            with torch.no_grad():  # Ensure gradients are not computed for this operation
                for branch in model.branches:
                    # branch.conv0.weight.clamp_(min=0)
                    branch.conv1.weight.clamp_(min=0)

            total_train_loss += total_batch_loss.item()  

        train_loss = total_train_loss / len(train_dataloader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_dataloader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs, _, _ = model(X_batch)
                # val_loss = criterion(outputs, Y_batch)
                val_loss = mse_loss(outputs, Y_batch)
                total_val_loss += val_loss.item()
        
        val_loss = total_val_loss / len(val_dataloader)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    return model, np.array(train_losses), np.array(val_losses)

def predict(model, X, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for prediction, saves memory and computations
        predictions, _, _ = model(X.to(device))
    return predictions.cpu()  # Move predictions to CPU

def calculate_correlation(Y_true, Y_pred):
    # Flatten both the true values and predictions
    Y_true_flat = Y_true.flatten()
    Y_pred_flat = Y_pred.flatten()
    correlationP = pearsonr(Y_true_flat, Y_pred_flat)[0]
    correlationS = spearmanr(Y_true_flat, Y_pred_flat)[0]
    return correlationP, correlationS

# Train individual models for all cells
