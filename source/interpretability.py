import ast
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import umap

def plot_training_losses(training_loss_file):
    """plots the training losses over epochs from a given file

    args:
        training_loss_file (_type_): path to the file containing training loss data; \
                                     each line in the file should be a dictionary with keys 'epoch' and 'train_loss'
    """

    epochs = []
    losses = []

    with open(training_loss_file, "r") as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            epochs.append(data["epoch"])
            losses.append(data["train_loss"])

    plt.plot(epochs, losses, marker = "o")
    
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("training loss over epochs")

    plt.grid(True)

    plt.show()

def plot_performance_measures(training_metrics_file, evaluation_metrics_file, metric = "accuracy"):
    """plots the performance measures over epochs from given files

    args:
        training_metrics_file (str): path to the file containing training metrics data; \
                                     each line in the file should be a dictionary with keys 'epoch' and the specified metric
        evaluation_metrics_file (str): path to the file containing evaluation metrics data; \
                                       each line in the file should be a dictionary with keys 'epoch' and the specified metric
        metric (str, optional): the performance measure to plot; \
                                defaults to 'accuracy'
    """

    train_epochs = []
    train_metrics = []

    with open(training_metrics_file, "r") as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            train_epochs.append(data["epoch"])
            train_metrics.append(data[metric])

    plt.plot(train_epochs, train_metrics, marker = "o", label = "training")

    evaluation_epochs = []
    evaluation_metrics = []

    with open(evaluation_metrics_file, "r") as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            evaluation_epochs.append(data["epoch"])
            evaluation_metrics.append(data[metric])

    plt.plot(evaluation_epochs, evaluation_metrics, marker = "x", label = "evaluation")

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} over epochs")
    plt.legend()

    plt.grid(True)

    plt.show()

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

    reduced_embeddings = reducer.fit_transform(embeddings.numpy())
    cluster_labels = clusterer.fit_predict(embeddings.numpy())

    cluster_average_predictions = [probability_predictions[cluster_labels == c].mean().item() for c in range(k)]

    colors = [cluster_average_predictions[label] for label in cluster_labels]

    plt.figure(figsize = (8, 6))
    
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c = colors, cmap = "coolwarm", s = 20)
    plt.colorbar(scatter, label = "average prediction (tumor edge probability)")
    
    plt.title("UMAP of learned graph embeddings")
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")

    plt.show()