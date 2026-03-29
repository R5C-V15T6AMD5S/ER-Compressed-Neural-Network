"""
prepare_dataset.py — Creates a smaller subset of Fruits-360 for training.

Takes the first 20 classes alphabetically from the full Fruits-360 dataset
and copies them into a new folder: fruits-360-small/

Usage:
    python prepare_dataset.py --src ./fruits-360 --dst ./fruits-360-small
    python prepare_dataset.py --src ./fruits-360 --dst ./fruits-360-small --n_classes 20

After running this, point all other scripts to --data_dir ./fruits-360-small
"""

import os
import shutil
import argparse


def prepare_subset(src_dir, dst_dir, n_classes=20):
    """
    Copies the first n_classes (alphabetically) from both
    Training/ and Test/ splits into a new destination folder.
    """

    for split in ["Training", "Test"]:
        src_split = os.path.join(src_dir, split)
        dst_split = os.path.join(dst_dir, split)

        if not os.path.isdir(src_split):
            print(f"[Error] Could not find '{src_split}'.")
            print(f"        Make sure --src points to the Fruits-360 root folder.")
            return

        # Get all class folders, sorted alphabetically
        all_classes = sorted([
            d for d in os.listdir(src_split)
            if os.path.isdir(os.path.join(src_split, d))
        ])

        selected = all_classes[:n_classes]

        print(f"\n[{split}] Copying {len(selected)} classes...")
        for cls in selected:
            src_cls = os.path.join(src_split, cls)
            dst_cls = os.path.join(dst_split, cls)

            if os.path.exists(dst_cls):
                print(f"  ✓ Already exists: {cls}")
                continue

            shutil.copytree(src_cls, dst_cls)
            n_images = len(os.listdir(dst_cls))
            print(f"  ✓ {cls:<35} ({n_images} images)")

    # Summary
    print("\n" + "=" * 50)
    print(f"  Dataset ready: '{dst_dir}'")
    print(f"  Classes ({n_classes}):")
    final_classes = sorted(os.listdir(os.path.join(dst_dir, "Training")))
    for i, cls in enumerate(final_classes):
        n_train = len(os.listdir(os.path.join(dst_dir, "Training", cls)))
        n_test  = len(os.listdir(os.path.join(dst_dir, "Test",     cls)))
        print(f"    {i+1:>2}. {cls:<35} train={n_train}, test={n_test}")

    total_train = sum(
        len(os.listdir(os.path.join(dst_dir, "Training", c)))
        for c in final_classes
    )
    total_test = sum(
        len(os.listdir(os.path.join(dst_dir, "Test", c)))
        for c in final_classes
    )
    print(f"\n  Total training images : {total_train}")
    print(f"  Total test images     : {total_test}")
    print(f"\n  Run your project with:")
    print(f"    python main.py --data_dir {dst_dir}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a small Fruits-360 subset for the project"
    )
    parser.add_argument("--src",       type=str, default="./fruits-360",
                        help="Path to the full Fruits-360 dataset")
    parser.add_argument("--dst",       type=str, default="./fruits-360-small",
                        help="Where to create the subset")
    parser.add_argument("--n_classes", type=int, default=20,
                        help="Number of classes to include (default: 20)")
    args = parser.parse_args()

    print(f"\n[Setup] Creating {args.n_classes}-class subset")
    print(f"        Source : {args.src}")
    print(f"        Output : {args.dst}")

    os.makedirs(args.dst, exist_ok=True)
    prepare_subset(args.src, args.dst, args.n_classes)
