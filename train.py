from feature_engineering import load_data
from model import NCModel
import geoopt
import torch

data = load_data('/Users/arnavshah/Downloads/hnn/data/pubmed')
model = NCModel()
optimizer = geoopt.optim.RiemannianAdam(params=model.parameters(), lr=0.03, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=200,
    gamma=0.5
)

best_val_metrics = model.init_metric_dict()
best_test_metrics = None
best_emb = None

for epoch in range(5000):
    model.train()
    optimizer.zero_grad()
    embeddings = model.encode(data['features'], data['adj_train_norm'])
    train_metrics = model.compute_metrics(embeddings, data, 'train')
    train_metrics['loss'].backward()
    optimizer.step()
    lr_scheduler.step()
    model.eval()
    embeddings = model.encode(data['features'], data['adj_train_norm'])
    val_metrics = model.compute_metrics(embeddings, data, 'val')
    print(f"Loss at {epoch} is: {val_metrics}")

print(f"\n========RESULTS=========\n{model.compute_metrics(embeddings, data, 'test')}")