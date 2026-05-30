"""
person3_pruning.py — Manual weight pruning implemented from scratch with NumPy.
PERSON 3's responsibility.

Implements train → prune → fine-tune pipeline inspired by Han et al. Section 2.2.
No torch.nn.utils.prune — everything is manual.

Two pruning strategies:
    1. Global pruning    — removes lowest-magnitude weights across ALL layers
    2. Per-layer pruning — removes a fixed % of weights within each layer

Usage:
    python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path models/cnn_fruits \
                              --strategy global --amount 0.5
"""

import os
import argparse
import numpy as np

from person1_model import SimpleCNN
from person2_train import (
    load_model,
    save_model,
    load_dataset,
    get_batches,
    cross_entropy_loss,
    SGDMomentum,
    compute_accuracy,
)



def npz_path(path):
    #osiguraj da file ima ispravni .npz nastavak
    return path if path.endswith(".npz") else path + ".npz"


def read_num_classes(model_path):
    #read the number of output classes stored in a saved model file
    
    path = npz_path(model_path) #convert the provided model path to the corresponding .npz file path
    data = np.load(path, allow_pickle=True) #allow_pickle=True used bc some stored objects may not be plain NumPy arrays

    if "num_classes" in data:
        return int(data["num_classes"][0])

    if "classes" in data:
        return len(data["classes"])

    return 131 #default


def get_weight_array(layer):
    #return only prunable weights, not biases
    #convlayer uses filters
    #fclayer uses weights
    if hasattr(layer, "filters"):
        return layer.filters
    return layer.weights


def set_weight_array(layer, new_weights):
    #gleda storea li sloj tezine u filtere ili weights, dodjeli novi tenzor
    if hasattr(layer, "filters"):
        layer.filters = new_weights
    else:
        layer.weights = new_weights


def create_all_one_masks(model):
    #create binary masks with same shapes as every pruneable weight tensor
    #biases are intentionally not pruned
    masks = {}

    for i, layer in enumerate(model.get_trainable_layers()):
        weights = get_weight_array(layer)
        masks[i] = np.ones_like(weights, dtype=np.float32)

    return masks


def apply_masks(model, masks):
    #force pruned weights to stay exactly zero.
    for i, layer in enumerate(model.get_trainable_layers()):
        weights = get_weight_array(layer)
        weights *= masks[i]
        set_weight_array(layer, weights)


def count_nonzero_weights(model):
    #count active and pruned weights in the model and calculate sparsity
    total = 0
    nonzero = 0

    for layer in model.get_trainable_layers():
        weights = get_weight_array(layer)
        total += weights.size
        nonzero += np.count_nonzero(weights)

    zero = total - nonzero
    sparsity = 100.0 * zero / total if total > 0 else 0.0

    return {
        "total": total,
        "nonzero": nonzero,
        "zero": zero,
        "sparsity": sparsity,
    }



#PRUNING

def global_prune(model, amount):
    #pruning percentage
    if not 0.0 <= amount < 1.0:
        raise ValueError("amount must be between 0.0 and 1.0")

    layers = model.get_trainable_layers()
    masks = create_all_one_masks(model) #sve tezine aktivne

    all_abs_weights = []
    layer_shapes = []

    for layer in layers:
        weights = get_weight_array(layer)
        all_abs_weights.append(np.abs(weights).reshape(-1))
        layer_shapes.append(weights.shape) #spremi original za kasniju rekonstrukciju

    #svi slojevi skupa
    flat_abs = np.concatenate(all_abs_weights)
    total_weights = flat_abs.size #br tezina ukupno
    prune_count = int(total_weights * amount) #br tezina za maknut

    if prune_count == 0:
        return masks

    #find globally smallest weights
    prune_indices = np.argsort(flat_abs)[:prune_count]

    #create global pruning mask
    global_mask_flat = np.ones(total_weights, dtype=np.float32)
    global_mask_flat[prune_indices] = 0.0

    #vrati globalnu masku u originalne dimenzije slojeva
    start = 0
    for i, shape in enumerate(layer_shapes):
        size = np.prod(shape)
        masks[i] = global_mask_flat[start:start + size].reshape(shape)
        start += size

    apply_masks(model, masks)
    return masks


def per_layer_prune(model, amount):

    if not 0.0 <= amount < 1.0:
        raise ValueError("amount must be between 0.0 and 1.0")

    masks = create_all_one_masks(model)

    #zasebno slojeve
    for i, layer in enumerate(model.get_trainable_layers()):
        weights = get_weight_array(layer) #dohvati tenzor
        flat_abs = np.abs(weights).reshape(-1)
        total = flat_abs.size #ukupan br tezina u sloju
        prune_count = int(total * amount) #za maknuti

        if prune_count == 0:
            continue

        prune_indices = np.argsort(flat_abs)[:prune_count] #nadi najmanje

        #create layer mask
        flat_mask = np.ones(total, dtype=np.float32)
        flat_mask[prune_indices] = 0.0

        masks[i] = flat_mask.reshape(weights.shape) #vrati og oblik

    apply_masks(model, masks)
    return masks


def prune_model(model, strategy, amount):
    if strategy == "global":
        return global_prune(model, amount)

    if strategy == "per_layer":
        return per_layer_prune(model, amount)

    raise ValueError("strategy must be 'global' or 'per_layer'")



#FINE-TUNING


def fine_tune(model, X_train, y_train, X_test, y_test, masks, epochs, batch_size, lr, momentum):
    
    #SGDMomentum azurira tezine koristeci trenutni gradijent i komponentu brzine koja se temelji na prethodnim azuriranjima
    optimizer = SGDMomentum(model, lr=lr, momentum=momentum)

    #fine tune po svakoj epohi
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size, shuffle=True):
            probs = model.forward(X_batch) #izracunaj vjerojatnosti klasa za trenutni batch

            #pretvori predvidene vjerojatnosti u predikcije klasa
            #predvidena klasa je indeks s najvecom vjerojatnoscu
            preds = np.argmax(probs, axis=1)
            train_correct += int(np.sum(preds == y_batch)) #koliko je predikcija tocno u ovom batchu
            train_total += len(y_batch) #dodaj br uzoraka u ovom batchu

            #cross-entropy gubitak i njegov gradijent s obzirom na izlazne vjerojatnosti modela
            loss, d_probs = cross_entropy_loss(probs, y_batch)

            model.backward(d_probs)
            optimizer.step(model)

            #keep pruned weights at zero
            apply_masks(model, masks)

            total_loss += loss
            batches += 1

        
        avg_loss = total_loss / batches if batches > 0 else 0.0 #average loss za sve batcheve u epohi
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0 #training accuracy za ovu epohu
        test_acc = compute_accuracy(model, X_test, y_test, batch_size=batch_size) #test accuracy nakon epohe

        print(
            f"Fine-tune Epoch [{epoch}/{epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2f}% | "
            f"Test Accuracy: {test_acc:.2f}%"
        )

    return model



#MAIN

def run_pruning(args):
    np.random.seed(args.seed)

    print("=" * 70)
    print("PERSON 3 — MANUAL NUMPY PRUNING")
    print("=" * 70)

    model_file = npz_path(args.model_path)

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Trained model not found: {model_file}\n"
            "Run person2_train.py first."
        )

    num_classes = read_num_classes(args.model_path)

    print(f"\n[1] Loading model")
    print(f"Model path  : {npz_path(args.model_path)}")
    print(f"Num classes : {num_classes}")

    model = SimpleCNN(num_classes=num_classes) #create a new simpleCNN instance with the correct output size
    load_model(model, args.model_path) #load previously trained weights into the model

    print("\n[2] Loading dataset")
    #load training data, used only if fine-tuning is enabled
    X_train, y_train, classes = load_dataset(
        args.data_dir,
        split="Training",
        img_size=args.img_size,
        max_per_class=args.max_per_class,
    )

    #used to evaluate accuracy before and after pruning
    X_test, y_test, _ = load_dataset(
        args.data_dir,
        split="Test",
        img_size=args.img_size,
        max_per_class=args.max_per_class,
    )

    print("\n[3] Accuracy before pruning")
    before_acc = compute_accuracy(model, X_test, y_test, batch_size=args.batch_size) #compute baseline test accuracy before any weights are removed
    before_stats = count_nonzero_weights(model) #count total, non-zero, zero weights, and initial sparsity

    print(f"Test Accuracy : {before_acc:.2f}%")
    print(f"Total weights : {before_stats['total']:,}")
    print(f"Zero weights  : {before_stats['zero']:,}")
    print(f"Sparsity      : {before_stats['sparsity']:.2f}%")

    print("\n[4] Applying pruning")
    print(f"Strategy : {args.strategy}")
    print(f"Amount   : {args.amount * 100:.1f}%")

    #apply the selected pruning method
    masks = prune_model(model, args.strategy, args.amount)

    #evaluate model immediately after pruning, before fine-tuning
    after_prune_acc = compute_accuracy(model, X_test, y_test, batch_size=args.batch_size)
    after_stats = count_nonzero_weights(model) #how many weight wer pruned

    print("\nAfter pruning:")
    print(f"Test Accuracy : {after_prune_acc:.2f}%")
    print(f"Total weights : {after_stats['total']:,}")
    print(f"Pruned Weights: {after_stats['zero']:,}")
    print(f"Zero weights  : {after_stats['zero']:,}")
    print(f"Sparsity      : {after_stats['sparsity']:.2f}%")

    #opcionalni fine-tune
    if args.finetune_epochs > 0: #npr. --finetune_epochs 4
        print("\n[5] Fine-tuning pruned model")
        fine_tune(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            masks=masks,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
        )
    else:
        print("\n[5] Fine-tuning skipped")

    print("\n[6] Final evaluation")
    final_acc = compute_accuracy(model, X_test, y_test, batch_size=args.batch_size) #evaluate final model after pruning and optional fine-tuning
    final_stats = count_nonzero_weights(model) #recompute sparsity to confirm pruned weights remain zero

    print(f"Original Accuracy     : {before_acc:.2f}%")
    print(f"After Pruning Accuracy: {after_prune_acc:.2f}%")
    print(f"Final Accuracy        : {final_acc:.2f}%")
    print(f"Final Sparsity        : {final_stats['sparsity']:.2f}%")

    print(f"Accuracy Change After Pruning : {after_prune_acc - before_acc:.2f}%")
    print(f"Accuracy Change Final         : {final_acc - before_acc:.2f}%")

    print("\n[7] Saving pruned model")
    save_model(model, args.save_path, classes=classes)

    mask_path = npz_path(args.mask_path)
    #create the output directory for masks if it does not already exist
    os.makedirs(os.path.dirname(mask_path), exist_ok=True) if os.path.dirname(mask_path) else None
    np.savez(mask_path, **{f"mask_layer{i}": mask for i, mask in masks.items()})

    print(f"Pruned model saved to : {npz_path(args.save_path)}")
    print(f"Pruning masks saved to: {mask_path}")

    print("\nSUMMARY")
    print("-" * 40)
    print(f"Strategy       : {args.strategy}")
    print(f"Pruning Amount : {args.amount*100:.0f}%")
    print(f"Sparsity       : {final_stats['sparsity']:.2f}%")
    print(f"Final Accuracy : {final_acc:.2f}%")
    print("-" * 40)

    print("=" * 70)

    return model



#CLI, obrađuje argumente te pokrece pruning
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Person 3: Manual magnitude pruning for SimpleCNN using NumPy"
    )

    parser.add_argument("--data_dir", type=str, default="./fruits-360-100x100")
    parser.add_argument("--model_path", type=str, default="models/cnn_fruits")
    parser.add_argument("--save_path", type=str, default="models/cnn_fruits_pruned")
    parser.add_argument("--mask_path", type=str, default="models/pruning_masks")

    parser.add_argument(
        "--strategy",
        type=str,
        default="global",
        choices=["global", "per_layer"],
        help="Pruning strategy: global or per_layer",
    )

    parser.add_argument(
        "--amount",
        type=float,
        default=0.5,
        help="Fraction of weights to prune, e.g. 0.5 means 50%%",
    )

    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--img_size", type=int, default=100)
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    run_pruning(parser.parse_args())