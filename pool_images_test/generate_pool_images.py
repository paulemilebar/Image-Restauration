import random
from pathlib import Path
import shutil
from typing import List, Tuple

def create_benchmark(
        source_dir: str = "./BSDS300/images/test",
        out_dir: str = "./pool_images_test/benchmark_10_images",
        n_images: int = 10,
        seed: int = 42,
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
) -> List[str]:
    """
    Randomly select n_images among training images.
    """
    out_path = Path(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    source_path = Path(source_dir)
    all_paths = [
        str(p) for p in source_path.rglob("*") 
        if p.is_file() and p.suffix.lower() in exts
    ]

    if not all_paths:
        raise RuntimeError(f"No image found in '{source_dir}'.")
    if len(all_paths) < n_images:
        raise RuntimeError(f"Not enough images : {len(all_paths)} found, {n_images} asked.")

    rng = random.Random(seed)
    chosen_paths = rng.sample(all_paths, n_images)

    final_paths = []
    for img_path in chosen_paths:
        destination = out_path / Path(img_path).name
        
        if not destination.exists():
            shutil.copy2(img_path, destination)
            
        final_paths.append(str(destination))

    print(f"{len(final_paths)} images in '{out_path}'.")
    return final_paths


create_benchmark(n_images=10)