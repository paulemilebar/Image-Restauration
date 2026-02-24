import random
from pathlib import Path
import shutil
from typing import List, Tuple

def create_benchmark(
    source_dir: str = "./BSDS300/images/test",
    out_dir: str = "./BSDS300/images/images_benchmark/benchmark_10_images",
    n_images: int = 10,
    seed: int = 42,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
):
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


#create_benchmark(n_images=10)

def specific_image(
    source_dir: str, 
    ind_image: str,
    out_dir: str
):
    """
    Copy a specific image for visual benchmarks
    """
    src_path = Path(source_dir)
    dest_path = Path(out_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    found_image = None
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        search_pattern = f"**/{ind_image}{ext}"
        results = list(src_path.rglob(search_pattern))
        
        if results:
            found_image = results[0]
            break

    if found_image:
        destination = dest_path / found_image.name
        shutil.copy2(found_image, destination)
        print(f"Image {found_image.name} found and copied in {out_dir}")
        return str(destination)
    else:
        print(f"Image '{ind_image}' not found in {source_dir}")
        return None

# Image plane
specific_image(
    source_dir= "./BSDS300/images/test",  
    ind_image="37073", 
    out_dir="./BSDS300/images/images_benchmark/image_plane"
)

# Image castel
specific_image(
    source_dir= "./BSDS300/images/test",  
    ind_image="102061", 
    out_dir="./BSDS300/images/images_benchmark/image_castel"
)

