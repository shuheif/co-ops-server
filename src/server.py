from typing import Dict, List, Tuple
import flwr as fl
from flwr.common import ndarrays_to_parameters, Metrics
import torch
import argparse
from pathlib import Path
from texttable import Texttable

from train import ResNetClassifier

NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_config(server_round: int) -> Dict[str, str]:
    config = {
        "batch_size": 16,
        "current_round": server_round,
        "local_epochs": 1,
    }
    return config

def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    avg_accuracy = sum(accuracies) / sum(examples)
    return {"avg_accuracy": avg_accuracy}

def evaluate_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    avg_accuracy = sum(accuracies) / sum(examples)
    t = Texttable()
    t.add_rows([
        ['Test metric', 'Aggregated value'],
        ['Weighted average accuracy', f'{avg_accuracy:.16f}'],
    ])
    print(t.draw())
    return {"avg_accuracy": avg_accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-Ops Server")
    parser.add_argument("--init_weights", type=str, help="path to your .ckpt file")
    parser.add_argument("--enable_ssh", type=bool, default=False)
    args = parser.parse_args()
    if args.init_weights:
        model = ResNetClassifier.load_from_checkpoint(args.ckpt_path, map_location=DEVICE)
    else:
        model = ResNetClassifier(NUM_CLASSES)
    certificates = (
        Path(".cache/certificates/ca.crt").read_bytes(),
        Path(".cache/certificates/server.pem").read_bytes(),
        Path(".cache/certificates/server.key").read_bytes(),
    ) if args.enable_ssh else None
    model_parameters = ndarrays_to_parameters(val.cpu().numpy() for _, val in model.state_dict().items())
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1., # Sample 100% of available clients for training
        fraction_evaluate=1., # Sample 100% of available clients for evaluation
        on_fit_config_fn=fit_config,
        initial_parameters=model_parameters,
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_metrics_aggregation_fn=evaluate_weighted_average,
    )
    hist = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        certificates=certificates,
    )
    assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
