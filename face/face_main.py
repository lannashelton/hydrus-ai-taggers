"""
Face tagger for Hydrus – two‑step pipeline.
Step 1: detect faces (once)
Step 2: recognize persons (iterative, tunable)
"""

import os, sqlite3, tempfile, hashlib, click
from io import BytesIO
from pathlib import Path
from collections import Counter
from itertools import combinations
import numpy as np
from PIL import Image, ImageFile
import cv2, onnxruntime

from sklearn.neighbors import NearestNeighbors
from collections import Counter

from interrogate_faces import FaceInterrogator, _align_face   # needed for debug_align
import hydrus_api

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DISTANCE_METHOD = "cosine_similarity"   # default, overridden by --distance-method
DB_PATH = "face_embeddings.db"

# ----------------------------------------------------------------------
#  Database helpers
# ----------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT NOT NULL,
            face_index INTEGER NOT NULL,
            bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
            embedding BLOB NOT NULL,
            person_id TEXT,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        )
    """)
    conn.commit()
    conn.close()

def create_new_person():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COALESCE(MAX(CAST(SUBSTR(person_id,2) AS INTEGER)),0) FROM persons")
    new_id = f"p{c.fetchone()[0] + 1}"
    c.execute("INSERT INTO persons (person_id) VALUES (?)", (new_id,))
    conn.commit()
    conn.close()
    return new_id

def store_face(file_hash, face_idx, bbox, embedding, person_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO faces (file_hash, face_index, bbox_x1, bbox_y1, bbox_x2,
                           bbox_y2, embedding, person_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (file_hash, face_idx, bbox[0], bbox[1], bbox[2], bbox[3],
          embedding.tobytes(), person_id))
    conn.commit()
    conn.close()

def load_all_faces(only_unassigned=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if only_unassigned:
        c.execute("SELECT id, file_hash, embedding, person_id FROM faces WHERE person_id IS NULL")
    else:
        c.execute("SELECT id, file_hash, embedding, person_id FROM faces")
    rows = c.fetchall()
    conn.close()
    return [{'id':r[0], 'file_hash':r[1], 'emb':np.frombuffer(r[2], dtype=np.float32), 'person_id':r[3]} for r in rows]

def assign_face_to_person(face_id, person_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
    conn.commit()
    conn.close()

def get_file_tags():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT file_hash, person_id FROM faces WHERE person_id IS NOT NULL")
    file_tags = {}
    for fhash, pid in c.fetchall():
        file_tags.setdefault(fhash, set()).add(f"person:{pid}")
    conn.close()
    return file_tags

# ----------------------------------------------------------------------
#  Distance computation (global-aware)
# ----------------------------------------------------------------------
def distance(a, b, method=None):
    if method is None:
        method = DISTANCE_METHOD   # reads the current global value
    if method == "cosine_similarity":
        return 1.0 - np.dot(a, b)
    return np.linalg.norm(a - b)

# ----------------------------------------------------------------------
#  Clustering (live neighbours)
# ----------------------------------------------------------------------
def cluster_faces(all_faces, max_distance, min_faces, allow_new=True):
    if not all_faces:
        return 0

    n = len(all_faces)
    X = np.vstack([f['emb'] for f in all_faces])

    metric = 'cosine' if DISTANCE_METHOD == 'cosine_similarity' else 'euclidean'
    nbrs = NearestNeighbors(radius=max_distance, metric=metric, n_jobs=-1)
    nbrs.fit(X)

    # Retrieve all neighbours (each list includes the point itself)
    raw_neighbours = nbrs.radius_neighbors(X, return_distance=False)

    # Remove the face itself to match the original behaviour
    neighbour_indices = [np.setdiff1d(neigh, [i]) for i, neigh in enumerate(raw_neighbours)]

    # Degree = number of other faces within max_distance
    degree = np.array([len(neigh) for neigh in neighbour_indices])
    order = np.argsort(-degree)   # descending

    # Working state
    person_ids = [None] * n          # person_id assigned to each index
    assigned = 0

    for idx in order:
        if person_ids[idx] is not None:
            continue

        neighbours = neighbour_indices[idx]

        # Which neighbours already have a person?
        existing = [person_ids[j] for j in neighbours if person_ids[j] is not None]
        if existing:
            # Use the most frequent person among assigned neighbours
            best_pid = Counter(existing).most_common(1)[0][0]
        else:
            # Core point condition: at least min_faces other faces nearby
            if len(neighbours) >= min_faces and allow_new:
                best_pid = create_new_person()
            else:
                # Not a core point, leave for later
                continue

        # Assign this face
        assign_face_to_person(all_faces[idx]['id'], best_pid)
        person_ids[idx] = best_pid
        all_faces[idx]['person_id'] = best_pid   # in‑memory update for later neighbours
        assigned += 1

    return assigned

# ----------------------------------------------------------------------
#  Hydrus helpers
# ----------------------------------------------------------------------
def get_tag_service_key(client, service_name):
    services = client.get_services()
    for s in services.get('local_tags', []):
        if s['name'] == service_name: return s['service_key']
    for s in services.get('all_known_tags', []):
        if s['name'] == service_name: return s['service_key']
    raise ValueError(f"Tag service '{service_name}' not found.")

def push_tags(client, tag_service, extra_tags=None):
    file_tags = get_file_tags()
    if not file_tags: return
    key = get_tag_service_key(client, tag_service)
    for fhash, tags in file_tags.items():
        tag_list = list(tags)
        if extra_tags: tag_list.extend(extra_tags)
        try:
            client.add_tags(hashes=[fhash], service_keys_to_tags={key: tag_list})
        except Exception as e:
            click.echo(f"Failed to tag {fhash}: {e}")

# ----------------------------------------------------------------------
#  Model loader
# ----------------------------------------------------------------------
def find_model_paths():
    candidates = [Path("./face/model"), Path("../model")]
    det_names = ["scrfd_2.5g_kps.onnx"]
    rec_names = ["w600k_r50.onnx", "arcface_resnet100.onnx"]
    det_path = rec_path = None
    for base in candidates:
        if not det_path:
            for n in det_names:
                p = base / n
                if p.is_file(): det_path = str(p); break
        if not rec_path:
            for n in rec_names:
                p = base / n
                if p.is_file(): rec_path = str(p); break
    if not det_path or not rec_path:
        raise FileNotFoundError("Models not found in ./model/ or ../model/.")
    return det_path, rec_path

class FaceTagInterrogator:
    def __init__(self, device="CPU", threshold=0.6):
        self.device, self.threshold, self.interrogator = device, threshold, None
    def load(self):
        det, rec = find_model_paths()
        self.interrogator = FaceInterrogator(det, rec, device=self.device)
    def interrogate(self, image: Image) -> dict:
        if not self.interrogator: raise RuntimeError("Model not loaded")
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dets = self.interrogator.detect_faces(img_np, conf=self.threshold)
        # InsightFace already provides embeddings – just extract them
        return {'face_count': len(dets), 'faces': dets}

def extract_video_frames(video_bytes, num_frames=30):
    frames = []
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as tmp:
        tmp.write(video_bytes); tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened(): return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); return frames
        indices = np.linspace(0, total-1, min(num_frames, total), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
    return frames

# ----------------------------------------------------------------------
#  CLI commands
# ----------------------------------------------------------------------
@click.group()
def cli(): pass

@click.command()
@click.argument("filename")
@click.option("--device", default="CPU")
@click.option("--threshold", default=0.6)
def evaluate(filename, device, threshold):
    """Evaluate a single image."""
    interrogator = FaceTagInterrogator(device=device, threshold=threshold)
    interrogator.load()
    img = Image.open(filename).convert("RGB")
    result = interrogator.interrogate(img)
    click.echo(f"Found {result['face_count']} faces.")
    for i, det in enumerate(result["faces"]):
        x1,y1,x2,y2 = det['bbox']
        click.echo(f"  Face {i}: ({x1},{y1})-({x2},{y2}) score {det['score']:.3f}")

@click.command()
@click.argument("filename")
@click.option("--device", default="CPU")
@click.option("--output", default="debug_faces")
def debug_align(filename, device, output):
    """Save detected faces with landmarks and aligned faces to disk."""
    import matplotlib.pyplot as plt
    import os
    interrogator = FaceTagInterrogator(device=device)
    interrogator.load()
    img = Image.open(filename).convert("RGB")
    img_np = np.array(img)
    result = interrogator.interrogate(img)
    dets = result['faces']

    os.makedirs(output, exist_ok=True)
    for i, det in enumerate(dets):
        # Draw landmarks on original
        vis = img_np.copy()
        for pt in det['landmarks']:
            cv2.circle(vis, tuple(pt.astype(int)), 2, (0, 255, 0), -1)
        cv2.rectangle(vis, (det['bbox'][0], det['bbox'][1]),
                     (det['bbox'][2], det['bbox'][3]), (0, 0, 255), 2)

        # Aligned face (requires _align_face)
        aligned_bgr = _align_face(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), det)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(vis); ax1.set_title("Landmarks + box")
        ax2.imshow(aligned_rgb); ax2.set_title("Aligned 112×112")
        plt.savefig(os.path.join(output, f"face_{i}.png"))
        plt.close()
        click.echo(f"Saved {os.path.join(output, f'face_{i}.png')}")

@click.command()
@click.argument("hashfile")
@click.option("--token", help="Hydrus API token")
@click.option("--device", default="CPU")
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--tag-service", default="A.I. Tags")
@click.option("--threshold", default=0.6)
def detect_batch(hashfile, token, device, host, tag_service, threshold):
    """
    DETECT faces only – store embeddings, tag files with 'ai face detected'.
    Does NOT create person tags.
    """
    if not os.path.isfile(hashfile): raise ValueError("hashfile not found")
    init_db()
    interrogator = FaceTagInterrogator(device=device, threshold=threshold)
    interrogator.load()
    client = hydrus_api.Client(token, host)

    with open(hashfile) as f:
        hashes = [line.strip() for line in f if line.strip()]
    if not hashes: return

    try:
        metadata = client.get_file_metadata(hashes=hashes)
        mime_map = {m['hash']: m['mime'] for m in metadata['metadata']}
    except Exception as e:
        click.echo(f"Failed to fetch metadata: {e}"); return

    total = 0
    with click.progressbar(hashes, label="Detecting faces") as bar:
        for fhash in bar:
            try:
                mime = mime_map.get(fhash, "")
                if "video/" in mime:
                    resp = client.get_file(fhash)
                    frames = extract_video_frames(resp.content)
                    faces = []
                    for frame in frames:
                        faces.extend(interrogator.interrogate(frame)['faces'])
                else:
                    img = Image.open(BytesIO(client.get_file(fhash).content)).convert("RGB")
                    faces = interrogator.interrogate(img)['faces']
                for i, det in enumerate(faces):
                    store_face(fhash, i, det['bbox'], det['embedding'])
                    total += 1
                # Tag the file as "ai face detected" (only once per file)
                if faces:
                    try:
                        key = get_tag_service_key(client, tag_service)
                        client.add_tags(hashes=[fhash],
                                        service_keys_to_tags={key: ["ai face detected"]})
                    except Exception as e:
                        click.echo(f"Failed to tag {fhash}: {e}")
            except Exception as e:
                click.echo(f"Error {fhash}: {e}")
    click.echo(f"Detected {total} faces. Embeddings stored. Tags added.")
    # Diagnostic
    #all_faces = load_all_faces()
    #if len(all_faces) > 1:
        #dists = [distance(f['emb'], g['emb']) for f,g in combinations(all_faces,2)]
        #click.echo(f"Pairwise distances (method={DISTANCE_METHOD}): min={min(dists):.3f} median={np.median(dists):.3f} max={max(dists):.3f} mean={np.mean(dists):.3f}")

@click.command()
@click.option("--token", help="Hydrus API token")
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--tag-service", default="A.I. Tags")
@click.option("--max-distance", default=0.65, help="Neighbour distance threshold")
@click.option("--min-faces", default=3, help="Core point threshold")
@click.option("--allow-new/--no-new", default=True, help="Allow creation of new persons")
@click.option("--distance-method", default="cosine_similarity", type=click.Choice(["euclidean","cosine_similarity"]))
def recognize(token, host, tag_service, max_distance, min_faces, allow_new, distance_method):
    """RECOGNIZE persons – cluster stored embeddings and push person:p# tags."""
    global DISTANCE_METHOD
    DISTANCE_METHOD = distance_method
    init_db()
    all_faces = load_all_faces()
    if not all_faces:
        click.echo("No faces in database."); return
    assigned = cluster_faces(all_faces, max_distance, min_faces, allow_new=allow_new)
    click.echo(f"Assigned {assigned} faces.")
    client = hydrus_api.Client(token, host)
    push_tags(client, tag_service, extra_tags=["face ai generated tags"])
    click.echo("Tags pushed.")

@click.command()
@click.option("--token", help="Hydrus API token")
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--tag-service", default="A.I. Tags")
@click.option("--distance-method", default="cosine_similarity",
              type=click.Choice(["euclidean","cosine_similarity"]))
@click.option("--max-distance", default=0.5, type=float)
@click.option("--stages", default="20,5,3,1",
              help="Comma‑separated list of min_faces values")
@click.option("--allow-new/--no-new", default=True)
def recognize_staged(token, host, tag_service, distance_method, max_distance, stages, allow_new):
    """Run recognition in multiple stages, from highest to lowest min_faces."""
    min_faces_list = [int(x.strip()) for x in stages.split(",")]

    global DISTANCE_METHOD
    DISTANCE_METHOD = distance_method

    init_db()
    for min_faces in min_faces_list:
        click.echo(f"Stage: min_faces={min_faces}, max_distance={max_distance}")
        all_faces = load_all_faces()
        assigned = cluster_faces(all_faces, max_distance, min_faces, allow_new=allow_new)
        click.echo(f"  Assigned {assigned} faces")

        # Push after each stage so you can visually check progress
        if token:
            client = hydrus_api.Client(token, host)
            push_tags(client, tag_service)

    click.echo("Staged recognition complete.")

@click.command()
def reset():
    """Clear all person assignments but keep face embeddings."""
    import sqlite3 as sql
    conn = sql.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM persons")
    c.execute("UPDATE faces SET person_id = NULL")
    conn.commit()
    conn.close()
    click.echo("All person assignments cleared. Embeddings are kept.")

@click.command()
@click.option("--person-ids", multiple=True, help="Person IDs to analyse (e.g. --person-ids p1 --person-ids p2)")
@click.option("--sample-size", default=10, help="Max faces to compare per person")
def diagnose_distances(person_ids, sample_size):
    """
    Show distance distributions for known persons (after manual assignment).
    Compare intra‑person vs inter‑person distances.
    """
    import random
    random.seed(42)
    init_db()
    all_faces = load_all_faces()
    if not person_ids:
        click.echo("Please provide at least one --person-ids (e.g. --person-ids p1)")
        return

    intra_dists = []
    inter_dists = []

    # Collect embeddings per person
    person_embs = {}
    for pid in person_ids:
        embs = [f['emb'] for f in all_faces if f['person_id'] == pid]
        if not embs:
            click.echo(f"No faces found for person {pid}")
            continue
        if len(embs) > sample_size:
            embs = random.sample(embs, sample_size)
        person_embs[pid] = embs

    # Intra‑person distances
    for pid, embs in person_embs.items():
        n = len(embs)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i+1, n):
                d = distance(embs[i], embs[j])
                intra_dists.append(d)

    # Inter‑person distances (between different persons)
    pids = list(person_embs.keys())
    for i in range(len(pids)):
        for j in range(i+1, len(pids)):
            for emb_a in person_embs[pids[i]]:
                for emb_b in person_embs[pids[j]]:
                    d = distance(emb_a, emb_b)
                    inter_dists.append(d)

    if intra_dists:
        click.echo(f"Intra‑person distances ({len(intra_dists)} pairs): "
                   f"min={min(intra_dists):.3f}, median={np.median(intra_dists):.3f}, "
                   f"max={max(intra_dists):.3f}, mean={np.mean(intra_dists):.3f}")
    if inter_dists:
        click.echo(f"Inter‑person distances ({len(inter_dists)} pairs): "
                   f"min={min(inter_dists):.3f}, median={np.median(inter_dists):.3f}, "
                   f"max={max(inter_dists):.3f}, mean={np.mean(inter_dists):.3f}")

    if intra_dists and inter_dists:
        overlap = sum(1 for d in inter_dists if d < max(intra_dists))
        click.echo(f"Overlap: {overlap} inter‑person pairs are closer than the *worst* intra‑person pair.")
        # Suggested threshold: somewhere between median intra and median inter
        suggested = (np.median(intra_dists) + np.median(inter_dists)) / 2
        click.echo(f"Suggested max_distance ({DISTANCE_METHOD}): {suggested:.2f}")
    elif intra_dists:
        suggested = max(intra_dists) + 0.05
        click.echo(f"No inter‑person data. Based on intra only, suggested max_distance: {suggested:.2f}")

@click.command()
@click.option("--token", help="Hydrus API token")
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--tag-service", default="A.I. Tags")
def recluster(token, host, tag_service):
    """Global DBSCAN re‑cluster (scikit‑learn)."""
    from sklearn.cluster import DBSCAN
    init_db()
    faces = load_all_faces()
    if not faces: click.echo("No faces."); return
    X = np.vstack([f['emb'] for f in faces])
    clustering = DBSCAN(eps=0.25, min_samples=3, metric='cosine').fit(X)
    labels = clustering.labels_
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM persons")
    conn.execute("UPDATE faces SET person_id = NULL")
    conn.commit(); conn.close()
    label2pid = {}
    for face, label in zip(faces, labels):
        if label == -1:
            pid = create_new_person()
        elif label not in label2pid:
            pid = create_new_person()
            label2pid[label] = pid
        else:
            pid = label2pid[label]
        assign_face_to_person(face['id'], pid)
    file_tags = get_file_tags()
    client = hydrus_api.Client(token, host)
    key = get_tag_service_key(client, tag_service)
    for fhash, tags in file_tags.items():
        client.add_tags(hashes=[fhash], service_keys_to_tags={key: list(tags) + ["face ai generated tags"]})
    click.echo("Re‑clustering complete.")

@click.command()
@click.option("--token", help="Hydrus API token")
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without actually deleting")
def clean(token, host, dry_run):
    """
    Remove faces for files that are deleted or trashed in Hydrus.
    Also removes empty person entries after cleanup.
    """
    import sqlite3 as sql

    client = hydrus_api.Client(token, host)
    conn = sql.connect(DB_PATH)
    c = conn.cursor()

    # 1. Collect all distinct hashes from the faces table
    c.execute("SELECT DISTINCT file_hash FROM faces")
    db_hashes = [row[0] for row in c.fetchall()]
    if not db_hashes:
        click.echo("No faces in database.")
        conn.close()
        return

    # 2. Query Hydrus in batches – full metadata to get is_deleted
    BATCH_SIZE = 256
    existing_hashes = set()
    total_db = len(db_hashes)

    # Optionally add a display message about size
    with click.progressbar(length=total_db, label="Checking Hydrus") as bar:
        for i in range(0, total_db, BATCH_SIZE):
            batch = db_hashes[i:i + BATCH_SIZE]
            try:
                # Default metadata (no only_return_*) includes is_deleted
                metadata = client.get_file_metadata(
                    hashes=batch,
                    include_services_object=False   # reduce overhead
                )
                for item in metadata.get("metadata", []):
                    # A hash is alive only if is_deleted is explicitly False
                    if item.get("is_deleted") is False:
                        existing_hashes.add(item["hash"])
            except Exception as e:
                click.echo(f"Warning: batch query failed: {e}")
            bar.update(len(batch))

    # 3. Determine missing hashes
    missing_hashes = set(db_hashes) - existing_hashes
    click.echo(f"\nFound {len(missing_hashes)} orphaned hashes out of {total_db} total.")

    if dry_run:
        if missing_hashes:
            click.echo("Would delete these hashes (first 10 shown):")
            for h in list(missing_hashes)[:10]:
                click.echo(f"  {h}")
            if len(missing_hashes) > 10:
                click.echo(f"  ... and {len(missing_hashes)-10} more.")
        conn.close()
        return

    # 4. Delete orphaned faces and empty persons
    for h in missing_hashes:
        c.execute("DELETE FROM faces WHERE file_hash = ?", (h,))
    c.execute("""
        DELETE FROM persons
        WHERE person_id NOT IN (
            SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    click.echo(f"Removed {len(missing_hashes)} orphaned faces and cleaned up empty persons.")

@click.command()
@click.option("--token", help="Hydrus API token")
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--device", default="CPU")
@click.option("--tag-service", default="A.I. Tags")
@click.option("--threshold", default=0.6)
@click.option("--limit", default=100, help="Maximum files per batch (and batch size for --full-auto)")
@click.option("--dry-run", is_flag=True, help="Only show which hashes would be processed")
@click.option("--full-auto", is_flag=True, help="Run continuously until all unprocessed files are exhausted")
def auto_detect(token, host, device, tag_service, threshold, limit, dry_run, full_auto):
    """
    Automatically search for files without 'ai face detected' or 'face not visible' tags
    and detect faces on them.

    In single-run mode, processes up to --limit files.
    In --full-auto mode, fetches and processes batches of --limit files repeatedly
    until no more unprocessed files exist (Ctrl+C to stop early).
    """
    if not dry_run:
        init_db()
        interrogator = FaceTagInterrogator(device=device, threshold=threshold)
        interrogator.load()
    else:
        interrogator = None  # not needed

    client = hydrus_api.Client(token, host)

    tag_to_exclude = [
        "-ai face detected",
        "-face not visible"
    ]

    while True:
        tags = tag_to_exclude + [f"system:limit={limit}"]
        try:
            result = client.search_files(
                tags=tags,
                return_hashes=True,
                return_file_ids=False
            )
            hashes = result.get("hashes", [])
        except Exception as e:
            click.echo(f"Search failed: {e}")
            break

        if not hashes:
            if full_auto:
                click.echo("No more files to process. Exiting full-auto mode.")
            else:
                click.echo("No new files to process.")
            break

        if dry_run:
            click.echo(f"Would process {len(hashes)} files (hashes):")
            for h in hashes:
                click.echo(h)
            if not full_auto:
                break    # single shot dry run
            # In full auto dry run, we shouldn't loop indefinitely? User would kill.
            # We can just exit after showing one batch to be safe.
            # But better to let them see all batches? Not needed. We'll just exit after one batch.
            # Actually, for full auto dry run we can just show the first batch to avoid infinite loop.
            # I'll break after one batch even in full auto dry run.
            break

        # Normal processing mode
        click.echo(f"Batch: {len(hashes)} files to process.")
        total_faces = 0
        tag_service_key = None

        with click.progressbar(hashes, label="Detecting faces") as bar:
            for fhash in bar:
                try:
                    # Get mime type
                    try:
                        meta = client.get_file_metadata(
                            hashes=[fhash],
                            only_return_basic_information=True
                        )
                        mime = meta["metadata"][0].get("mime", "")
                    except Exception:
                        mime = ""

                    if "video/" in mime:
                        resp = client.get_file(fhash)
                        frames = extract_video_frames(resp.content)
                        faces = []
                        for frame in frames:
                            faces.extend(interrogator.interrogate(frame)['faces'])
                    else:
                        img = Image.open(BytesIO(client.get_file(fhash).content)).convert("RGB")
                        faces = interrogator.interrogate(img)['faces']

                    if faces:
                        for i, det in enumerate(faces):
                            store_face(fhash, i, det['bbox'], det['embedding'])
                            total_faces += 1
                        tag_to_add = "ai face detected"
                    else:
                        tag_to_add = "face not visible"

                    if tag_service_key is None:
                        tag_service_key = get_tag_service_key(client, tag_service)
                    client.add_tags(
                        hashes=[fhash],
                        service_keys_to_tags={tag_service_key: [tag_to_add]}
                    )
                except Exception as e:
                    click.echo(f"Error {fhash}: {e}")

        click.echo(f"Batch done. Detected {total_faces} faces.")

        if not full_auto:
            break   # single run
    
    click.echo("auto_detect finished.")

cli.add_command(reset)
cli.add_command(evaluate)
cli.add_command(detect_batch)
cli.add_command(recognize)
cli.add_command(recluster)
cli.add_command(diagnose_distances)
cli.add_command(recognize_staged)
cli.add_command(debug_align)
cli.add_command(clean)
cli.add_command(auto_detect)


if __name__ == '__main__':
    Image.init()
    cli()
