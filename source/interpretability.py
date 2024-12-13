import ast
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import umap

def plot_training_losses(training_loss_file, ax = None):
    """plots the training losses over epochs from a given file

    args:
        training_loss_file (str): path to the file containing training loss data; \
                                  each line in the file should be a dictionary with keys 'epoch' and 'train_loss'
        ax (axes, optional): the axes on which to plot the figure; \
                             defaults to None, in which case a new figure and axes will be created
    """

    epochs = []
    losses = []

    with open(training_loss_file, "r") as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            epochs.append(data["epoch"])
            losses.append(data["train_loss"])

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(epochs, losses, marker = "o")
    
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("training loss over epochs")

    ax.grid(True)

    return ax

def plot_joint_training_losses(training_loss_files, model_types):
    """plots the training losses for multiple models on the same plot

    args:
        model_types (list): list of model types
        training_loss_files (list): list of file paths containing training loss data for each model
    """

    _, ax = plt.subplots()

    for training_loss_file in training_loss_files:
        plot_training_losses(training_loss_file, ax = ax)

    ax.legend(model_types)

    plt.savefig(f"images/joint_training_loss.png")

    plt.show()

def plot_performance_measures(training_metrics_file, evaluation_metrics_file, model_type, metric = "accuracy", ax = None):
    """plots the performance measures over epochs from given files

    args:
        training_metrics_file (str): path to the file containing training metrics data; \
                                     each line in the file should be a dictionary with keys 'epoch' and the specified metric
        evaluation_metrics_file (str): path to the file containing evaluation metrics data; \
                                       each line in the file should be a dictionary with keys 'epoch' and the specified metric
        model_type (str): the type of model being evaluated
        metric (str, optional): the performance measure to plot; \
                                defaults to 'accuracy'
        ax (axes, optional): the axes on which to plot the figure; \
                             defaults to None, in which case a new figure and axes will be created
    """

    train_epochs = []
    train_metrics = []

    with open(training_metrics_file, "r") as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            train_epochs.append(data["epoch"])
            train_metrics.append(data[metric])

    evaluation_epochs = []
    evaluation_metrics = []

    with open(evaluation_metrics_file, "r") as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            evaluation_epochs.append(data["epoch"])
            evaluation_metrics.append(data[metric])

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(train_epochs, train_metrics, marker = "o", label = "training")
    ax.plot(evaluation_epochs, evaluation_metrics, marker = "x", label = "evaluation")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{model_type} {metric} over epochs")
    ax.legend()

    ax.grid(True)

    return ax

def extract_embeddings(model, loader, device):
    """extracts embeddings, probability predictions, and labels from a given model and dataloader

    args:
        model (module): the model used to extract embeddings and make predictions
        loader (dataloader): the dataloader providing the batches of data
        device (device): the device (cpu or gpu) on which the computations will be performed
    """

    model.eval()

    embeddings = []
    probability_predictions = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            embedding = model.extract_embedding(batch.x, batch.edge_index, batch.batch)
            embeddings.append(embedding)

            logits = model(batch.x, batch.edge_index, batch.batch)
            predicted_probabilities = torch.softmax(logits, dim = 1)[:, 1]
            probability_predictions.append(predicted_probabilities)

            labels.append(batch.y.cpu())
    
    embeddings = torch.cat(embeddings, dim = 0)
    probability_predictions = torch.cat(probability_predictions, dim = 0)
    labels = torch.cat(labels, dim = 0)

    return embeddings, probability_predictions, labels

def visualize_embeddings(embeddings, probability_predictions, k):
    """visualize the embeddings using UMAP for dimensionality reduction and kmeans for clustering

    args:
        embeddings (tensor): the high-dimensional embeddings to be visualized
        probability_predictions (tensor): the probability predictions associated with each embedding
        k (int): the number of clusters to form
    """
    
    reducer = umap.UMAP(n_components = 2)
    clusterer = KMeans(n_clusters = k, random_state = 42)

    reduced_embeddings = reducer.fit_transform(embeddings.cpu().numpy())
    cluster_labels = clusterer.fit_predict(embeddings.cpu().numpy())

    cluster_average_predictions = [probability_predictions[cluster_labels == c].mean().item() for c in range(k)]

    colors = [cluster_average_predictions[label] for label in cluster_labels]

    plt.figure(figsize = (8, 6))
    
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c = colors, cmap = "coolwarm", s = 20)
    plt.colorbar(scatter, label = "average prediction (tumor edge probability)")
    
    plt.title("UMAP of learned graph embeddings")
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")

    plt.show()