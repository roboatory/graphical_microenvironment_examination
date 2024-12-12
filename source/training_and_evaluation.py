import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
                                recall_score, roc_auc_score
    
def compute_metrics(predictions, labels, average = "macro"):
    """computes various evaluation metrics for the given predictions and labels

    args:
        predictions (array): predicted labels
        labels (array): true labels
        average (str, optional): type of averaging performed on the data; \
                                 defaults to 'macro'
    """

    f1 = f1_score(labels, predictions, average = average)
    precision = precision_score(labels, predictions, average = average)
    recall = recall_score(labels, predictions, average = average)
    auc = roc_auc_score(labels, predictions, average = average, multi_class = "ovr")
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "accuracy": accuracy
    }

def train_epoch(model, loader, optimizer, criterion, device):
    """trains the model for one epoch

    args:
        model (module): the model to train
        loader (dataloader): the dataloader for training data
        optimizer (optimizer): the optimizer used for training
        criterion (module): the loss function
        device (device): the device to run the training on
    """
    
    model.train()

    cumulative_loss = 0
    predictions = []
    labels = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()

        predictions.extend(out.argmax(dim = 1).cpu().numpy())
        labels.extend(batch.y.cpu().numpy())
    
    cumulative_loss /= len(loader)

    metrics = compute_metrics(predictions, labels)
    return cumulative_loss, metrics

def evaluate_epoch(model, loader, device):
    """evaluates the model for one epoch

    args:
        model (module): the model to evaluate
        loader (dataloader): the dataloader for evaluation data
        device (device): the device to run the evaluation on
    """

    model.eval()

    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)
            
            predictions.extend(out.argmax(dim = 1).cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

    metrics = compute_metrics(predictions, labels)
    return metrics