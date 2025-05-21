import os
from PIL import Image
import imagehash

def get_image_hashes(folder):
    hashes = {}
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("L").resize((128, 128))
            h = imagehash.phash(img)
            hashes[fname] = h
        except:
            continue
    return hashes

def find_matching_images(folder1, folder2, threshold=5):
    hashes1 = get_image_hashes(folder1)
    hashes2 = get_image_hashes(folder2)
    
    matches = []
    for name1, hash1 in hashes1.items():
        for name2, hash2 in hashes2.items():
            if hash1 - hash2 <= threshold:  # 哈希距离越小，越相似
                matches.append((name1, name2, hash1 - hash2))
    return matches

matches = find_matching_images("./MedicalExpert-1/images/0Normal", "./MedicalExpert-2/images/1Doubtful")
for m in matches:
    print(f"Match: {m[0]} ↔ {m[1]} (distance={m[2]})")
