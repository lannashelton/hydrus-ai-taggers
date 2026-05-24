import os
import tempfile
import time
import cv2
import click
import requests
from PIL import Image, ImageFile
import interrogate
import hydrus_api
from io import BytesIO
import json

# ----------------------------------------------------------------------
# Custom requests session that enforces a default timeout
# ----------------------------------------------------------------------
class TimeoutSession(requests.Session):
    """A requests Session with a hard-coded default timeout."""
    def __init__(self, timeout=30):
        super().__init__()
        self.timeout = timeout

    def request(self, method, url, **kwargs):
        # If no timeout explicitly passed, use the default
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        return super().request(method, url, **kwargs)


def get_tag_service_key(client, service_name):
    """Get service key for a tag service by its display name."""
    services = client.get_services()
    
    for service in services.get('local_tags', []):
        if service['name'] == service_name:
            return service['service_key']
    
    for service in services.get('all_known_tags', []):
        if service['name'] == service_name:
            return service['service_key']
    
    raise ValueError(f"Tag service '{service_name}' not found. Available services: {list(s['name'] for s in services.get('local_tags', []) + services.get('all_known_tags', []))}")

def find_model_path(model_name):
    """Look for model folder in local ./model/ first, then ../model/"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    local_path = os.path.join(script_dir, 'model', model_name, 'info.json')
    if os.path.isfile(local_path):
        return local_path
    
    parent_path = os.path.join(script_dir, '..', 'model', model_name, 'info.json')
    if os.path.isfile(parent_path):
        return parent_path
    
    raise ValueError(
        f"info.json not found for model '{model_name}'. Tried:\n"
        f"  {local_path}\n"
        f"  {parent_path}"
    )

Image.MAX_IMAGE_PIXELS = None

kaomojis = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
]

def extract_video_frames(video_bytes, num_frames=5):
    """Extract N evenly spaced frames from video bytes."""
    frames = []
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as tmpfile:
        tmpfile.write(video_bytes)
        tmpfile.flush()

        cap = cv2.VideoCapture(tmpfile.name)
        if not cap.isOpened():
            print("❌ OpenCV failed to open video file")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("❌ Video has no frames")
            cap.release()
            return frames

        if num_frames == 1:
            indices = [total_frames // 2]
        else:
            step = max(1, (total_frames - 1) / (num_frames - 1))
            indices = [int(i * step) for i in range(num_frames)]
            if len(indices) > 1:
                indices[-1] = total_frames - 1

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to read frame at index {idx}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frames.append(image)

        cap.release()

    return frames


def _load_interrogator(model, device):
    """Helper to create and load the interrogator."""
    model_path = find_model_path(model)
    if not os.path.isfile(model_path):
        raise ValueError(f"info.json not found in model folder! Searched at: {model_path}")

    with open(model_path) as json_f:
        modelinfo = json.load(json_f)

    interrogator = interrogate.WaifuDiffusionInterrogator(
        modelinfo['modelname'],
        modelinfo['modelfile'],
        modelinfo['tagsfile'],
        model,
        modelinfo['ratingsflag'],
        modelinfo['numberofratings'],
        repo_id=modelinfo['source'],
    )
    interrogator.load(device)
    return interrogator, modelinfo


# ----------------------------------------------------------------------
#  CLI group
# ----------------------------------------------------------------------
@click.group()
def cli():
    pass


# ----------------------------------------------------------------------
#  Evaluate (single local file)
# ----------------------------------------------------------------------
@click.command()
@click.argument("filename")
@click.option("--device", default="CPU", type=click.Choice(["CPU", "GPU", "NPU"]),
              help="Execution device")
@click.option("--model", default="wd-eva02-large-tagger-v3", help="Tagging model")
@click.option("--threshold", default=0.35, help="Confidence threshold")
def evaluate(filename, device, model, threshold):
    interrogator, modelinfo = _load_interrogator(model, device)
    image = Image.open(filename)
    ratings, tags = interrogator.interrogate(image)

    rating = "none"
    if modelinfo['ratingsflag']:
        ratings["none"] = 0.0
        for key in ratings.keys():
            if ratings[key] > ratings[rating]:
                rating = key

    clipped_tags = []
    for key in tags.keys():
        if tags[key] > threshold:
            clipped_tags.append(key)

    click.echo("rating: " + rating)
    click.echo("tags: " + ", ".join(clipped_tags))


# ----------------------------------------------------------------------
#  Evaluate single hash via API
# ----------------------------------------------------------------------
@click.command()
@click.argument("hash")
@click.option("--token", required=True, help="Hydrus API token")
@click.option("--device", default="CPU", type=click.Choice(["CPU", "GPU", "NPU"]),
              help="Execution device")
@click.option("--model", default="wd-eva02-large-tagger-v3")
@click.option("--threshold", default=0.35)
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--tag-service", default="A.I. Tags")
@click.option("--ratings-only", is_flag=True, default=False)
@click.option("--privacy", is_flag=True, default=True)
def evaluate_api(hash, token, device, model, threshold, host, tag_service, ratings_only, privacy):
    interrogator, modelinfo = _load_interrogator(model, device)

    if ratings_only and not modelinfo['ratingsflag']:
        raise ValueError("--ratings-only set, but model does not support ratings!")

    client = hydrus_api.Client(token, host)
    image_bytes = BytesIO(client.get_file(hash).content)
    image = Image.open(image_bytes)
    ratings, tags = interrogator.interrogate(image)

    rating = "none"
    if modelinfo['ratingsflag']:
        ratings["none"] = 0.0
        for key in ratings.keys():
            if ratings[key] > ratings[rating]:
                rating = key

    clipped_tags = []
    if not ratings_only:
        for key in tags.keys():
            if tags[key] > threshold:
                clipped_tags.append(key.replace("_", " ") if key not in kaomojis else key)

    if not privacy:
        click.echo("rating: " + rating)
        click.echo("tags: " + ", ".join(clipped_tags))

    if modelinfo['ratingsflag']:
        clipped_tags.append("rating:" + rating)
    if ratings_only:
        clipped_tags.append("ratings only " + modelinfo['modelname'] + " ai generated tags")
    else:
        clipped_tags.append(modelinfo['modelname'] + " ai generated tags")

    try:
        tag_service_key = get_tag_service_key(client, tag_service)
        client.add_tags(hashes=[hash],
                        service_keys_to_tags={tag_service_key: clipped_tags})
    except Exception as e:
        click.echo(f"❌ Failed to add tags: {e}")


# ----------------------------------------------------------------------
#  Batch evaluate from hashfile
# ----------------------------------------------------------------------
@click.command()
@click.argument("hashfile")
@click.option("--token", required=True)
@click.option("--device", default="CPU", type=click.Choice(["CPU", "GPU", "NPU"]))
@click.option("--model", default="wd-eva02-large-tagger-v3")
@click.option("--threshold", default=0.35)
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--tag-service", default="A.I. Tags")
@click.option("--ratings-only", is_flag=True, default=False)
@click.option("--privacy", is_flag=True, default=True)
def evaluate_api_batch(hashfile, token, device, model, threshold, host, tag_service, ratings_only, privacy):
    """Evaluate and tag files from a hashfile (images & videos)."""
    if not os.path.isfile(hashfile):
        raise ValueError("hashfile not found!")
    
    interrogator, modelinfo = _load_interrogator(model, device)
    if ratings_only and not modelinfo['ratingsflag']:
        raise ValueError("--ratings-only set, but model does not support ratings!")

    client = hydrus_api.Client(token, host)

    with open(hashfile) as f:
        hashes = [line.strip() for line in f if line.strip()]

    if not hashes:
        click.echo("📭 No hashes provided.")
        return

    # Prefetch MIME types
    try:
        metadata = client.get_file_metadata(hashes=hashes)
        mime_map = {}
        for m in metadata['metadata']:
            mime_map[m['hash']] = m.get('mime', '')
    except Exception as e:
        click.echo(f"❌ Failed to get metadata: {e}")
        return

    with click.progressbar(hashes, label="Processing files") as bar:
        for file_hash in bar:
            try:
                mime = mime_map.get(file_hash, "")
                if "video/" in mime:
                    file_bytes = client.get_file(file_hash).content
                    frames = extract_video_frames(file_bytes, num_frames=5)
                    if not frames:
                        raise Exception("No frames extracted")
                    all_ratings = {}
                    all_tags = {}
                    for frame in frames:
                        ratings, tags = interrogator.interrogate(frame)
                        for k, v in (ratings or {}).items():
                            if k not in all_ratings or v > all_ratings[k]:
                                all_ratings[k] = v
                        for k, v in tags.items():
                            if k not in all_tags or v > all_tags[k]:
                                all_tags[k] = v
                    ratings, tags = all_ratings, all_tags
                else:
                    image_bytes = BytesIO(client.get_file(file_hash).content)
                    image = Image.open(image_bytes)
                    ratings, tags = interrogator.interrogate(image)

                rating = "none"
                if modelinfo['ratingsflag']:
                    ratings["none"] = 0.0
                    for key in ratings.keys():
                        if ratings[key] > ratings[rating]:
                            rating = key

                clipped_tags = []
                if not ratings_only:
                    for key in tags.keys():
                        if tags[key] > threshold:
                            clipped_tags.append(key.replace("_", " ") if key not in kaomojis else key)

                if not privacy:
                    click.echo(f"⭐ {rating} | 🏷️ {', '.join(clipped_tags[:10])}...")

                if modelinfo['ratingsflag']:
                    clipped_tags.append("rating:" + rating)
                if ratings_only:
                    clipped_tags.append("ratings only " + modelinfo['modelname'] + " ai generated tags")
                else:
                    clipped_tags.append(modelinfo['modelname'] + " ai generated tags")

                tag_service_key = get_tag_service_key(client, tag_service)
                client.add_tags(
                    hashes=[file_hash],
                    service_keys_to_tags={tag_service_key: clipped_tags}
                )
            except Exception as e:
                click.echo(f"❌ Error {file_hash}: {e}")
                continue


# ----------------------------------------------------------------------
#  Full‑auto: process all files missing a marker tag
# ----------------------------------------------------------------------
@click.command("full-auto")
@click.option("--token", required=True, help="Hydrus API token")
@click.option("--device", default="CPU", type=click.Choice(["CPU", "GPU", "NPU"]))
@click.option("--model", default="wd-eva02-large-tagger-v3")
@click.option("--threshold", default=0.35)
@click.option("--host", default="http://127.0.0.1:45869")
@click.option("--tag-service", default="A.I. Tags")
@click.option("--ratings-only", is_flag=True, default=False)
@click.option("--privacy", is_flag=True, default=True)
@click.option("--limit", default=100, help="Batch size per search")
@click.option("--dry-run", is_flag=True, default=False, help="Only show what would be processed")
@click.option("--marker-tag", default="wd14 tags generated",
              help="Tag that marks a file as already processed")
def full_auto(token, device, model, threshold, host, tag_service, ratings_only, privacy, limit, dry_run, marker_tag):
    """
    Continuously tag all files that don't have the marker tag.
    Ctrl+C to stop early.
    """
    interrogator = None
    if not dry_run:
        interrogator, modelinfo = _load_interrogator(model, device)
        if ratings_only and not modelinfo['ratingsflag']:
            raise ValueError("--ratings-only set, but model does not support ratings!")
    else:
        model_path = find_model_path(model)
        with open(model_path) as f:
            modelinfo = json.load(f)

    # --- Create a custom session with a default timeout of 30 seconds ---
    session = TimeoutSession(timeout=30)
    client = hydrus_api.Client(token, host, session=session)  # <--- timeout enforced via session

    while True:
        tags = [f"-{marker_tag}", f"system:limit={limit}"]
        try:
            result = client.search_files(
                tags=tags,
                return_hashes=True,
                return_file_ids=False
            )
            hashes = result.get("hashes", [])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            click.echo(f"⏳ Search request timed out or connection error: {e}")
            time.sleep(5)
            continue
        except Exception as e:
            click.echo(f"Search failed: {e}")
            time.sleep(5)
            continue

        if not hashes:
            click.echo("No new files to process. Done.")
            break

        if dry_run:
            click.echo(f"Would process {len(hashes)} files:")
            for h in hashes:
                click.echo(h)
            if click.confirm("Continue dry‑run with next batch?", default=True):
                continue
            else:
                break

        click.echo(f"Batch: {len(hashes)} files")
        processed = 0

        with click.progressbar(hashes, label="Tagging") as bar:
            for file_hash in bar:
                try:
                    meta = client.get_file_metadata(hashes=[file_hash], only_return_basic_information=True)
                    mime = meta["metadata"][0].get("mime", "")

                    if "video/" in mime:
                        file_bytes = client.get_file(file_hash).content
                        frames = extract_video_frames(file_bytes, num_frames=5)
                        if not frames:
                            continue
                        all_ratings = {}
                        all_tags = {}
                        for frame in frames:
                            r, t = interrogator.interrogate(frame)
                            for k,v in (r or {}).items():
                                if k not in all_ratings or v > all_ratings[k]:
                                    all_ratings[k] = v
                            for k,v in t.items():
                                if k not in all_tags or v > all_tags[k]:
                                    all_tags[k] = v
                        ratings, tags = all_ratings, all_tags
                    else:
                        img_bytes = BytesIO(client.get_file(file_hash).content)
                        img = Image.open(img_bytes)
                        ratings, tags = interrogator.interrogate(img)

                    rating = "none"
                    if modelinfo['ratingsflag']:
                        ratings["none"] = 0.0
                        rating = max(ratings, key=ratings.get)

                    clipped_tags = []
                    if not ratings_only:
                        for key, conf in tags.items():
                            if conf > threshold:
                                clipped_tags.append(key.replace("_", " ") if key not in kaomojis else key)

                    if modelinfo['ratingsflag']:
                        clipped_tags.append("rating:" + rating)
                    if ratings_only:
                        clipped_tags.append("ratings only " + modelinfo['modelname'] + " ai generated tags")
                    else:
                        clipped_tags.append(modelinfo['modelname'] + " ai generated tags")

                    clipped_tags.append(marker_tag)

                    tag_service_key = get_tag_service_key(client, tag_service)
                    client.add_tags(
                        hashes=[file_hash],
                        service_keys_to_tags={tag_service_key: clipped_tags}
                    )
                    processed += 1
                except Exception as e:
                    click.echo(f"❌ {file_hash}: {e}")
                    continue

        click.echo(f"Processed {processed} files in this batch.")
        time.sleep(2)   # let Hydrus re‑index before next search


# ----------------------------------------------------------------------
#  Register commands
# ----------------------------------------------------------------------
if __name__ == '__main__':
    Image.init()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cli.add_command(evaluate)
    cli.add_command(evaluate_api)
    cli.add_command(evaluate_api_batch)
    cli.add_command(full_auto)
    cli()
