import torch
import json
import random

with open("fashion.json", "r") as f:
    dataset = [json.loads(line.strip()) for line in f]

positive_pairs = [(item["scene"], item["product"]) for item in dataset]

all_products = list(set([item["product"] for item in dataset]))
negative_pairs = []
for scene, _ in positive_pairs:
    random_product = random.choice(all_products)
    while (scene, random_product) in positive_pairs:
        random_product = random.choice(all_products)
    negative_pairs.append((scene, random_product))

pairs = [(scene, product, 1) for scene, product in positive_pairs] + \
        [(scene, product, 0) for scene, product in negative_pairs]

random.shuffle(pairs)
split = int(0.8 * len(pairs))
train_pairs = pairs[:split]
test_pairs = pairs[split:]
