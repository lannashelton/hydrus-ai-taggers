# Hydrus AI Taggers Kit
Hydrus AI Taggers Kit is a modified version of **Garbevoir/wd-e621-hydrus-tagger**.
This is a work in progress and its current state is 'works on my machine'. I made this for personal usage and putting it here in case it may be useful for someone. I've tuned it to work on my linux mini pc with intel cpu and I've only tested it with openvino and intel npu. If you are using a different operating system or a different hardware, it may or may not work. I didn't have chance to test it on other platforms as I don't own a mac or a nvidia gpu. I'm sharing these scripts here hoping maybe it may help a fellow hydrus network user. Scripts have bunch of commands I didn't list here but you can see them with --help flag.

## Features
Hydrus AI Taggers Kit currently has two parts:
1. Generating booru-style tags using SmilingWolf's WD14 AI models.
2. Generating and recognizing people using Insightface's face detection and recognition AI models.

It has a full-auto mode that automatically fetches a number of files from your Hydrus Network and processes them. When done it fetches another batch and continues processing until manually stopped. There's also manual processing option like original wd-e621-hydrus-tagger using hashes.txt file.

It can tag videos in addition to images.

It can use CPU, GPU or NPU (if you have drivers installed and your system can see your NPU properly).

#### WD14 AI Tagging
WD14 AI tagger can tag images and videos with SmilingWolf's WD14 AI tagging models.

It creates booru-like tags (1girl, solo, brown hair, sitting etc.).

When AI tagger encounters a video, it generates 5 images from evenly distributed parts of the video and tags them. It is possible to manually increase or decrease the amount of frames generated from each video. More frames increases accuracy but makes each video take longer to process.

#### Face Detection and Recognition
Face detection and recognition is a 2 step process. Detecting faces creates a DB file with embeddings. Recognition command creates persons by looking at these embeddings and calculating their similarity with all other faces in DB. It is possible to automatically use a staged calculation that improves accuracy.

If face recognition recognizes and creates a person, it tags those files with a tag like `person:p1`. If you know who that person is, you can manually create a tag sibling in hydrus and turn that generated person tags into desired tags.

Example: `person:p1` -> `person:bob marley`

## Installation

1. **Clone this repository and get into it**
```bash
git clone https://github.com/lannashelton/hydrus-ai-taggers
cd hydrus-ai-taggers
```

2. **Create a virtual environment to avoid possible package conflicts**
```bash
python -m venv .venv
```

**Activate virtual environment (Linux/Mac)**
```bash
source .venv/bin/activate # Linux/Mac
```

**Activate virtual environment (Windows)**
```bash
.venv\Scripts\activate # Windows
```

3. **Install Dependencies**

| Hardware                                  | Command                                    |
| ----------------------------------------- | ------------------------------------------ |
| CPU (uses CPU cores, no igpu, gpu or npu) | `pip install -r requirements-cpu.txt`      |
| Nvidia GPU                                | `pip install -r requirements-gpu.txt`      |
| Intel Openvino (for igpu and npu)         | `pip install -r requirements-openvino.txt` |

For Nvidia GPU, make sure you have CUDA toolkit installed.

For Intel NPU make sure you have Intel NPU drivers and Level Zero api installed.

4. **Download WD14 AI Model**

Download *wd-eva02-large-tagger-v3* WD14 AI model from SmilingWolf's Huggingface page [here](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)

You only need to download *model.onnx*

5. **Put AI Model onnx in model folder**

Go to *wd14/model/wd-eva02-large-tagger-v3* folder. You will see *info.json* and *selected_tags.csv* files in that folder. Put the *model.onnx* file you downloaded into that location.

6. **Download face detection and face recognition onnx models.**
Download [scrfd_2.5g_kps.onnx](https://github.com/laolaolulu/FaceTrain/blob/master/model/scrfd/scrfd_2.5g_kps.onnx) face detection AI model.
Download [buffalo_l.zip](https://github.com/deepinsight/insightface/releases) AI model package.

7. **Put face detection and face recognition models in model folder**
Go to *face/model* folder. Put contents of *buffalo_l.zip* and *scrfd_2.5g_kps.onnx* in that folder.

8. **Prepare your Hydrus Network for AI tags**

If you didn't use Hydrus api before, you will have to enable it and create an api key so this scripts can communicate with your Hydrus Network.

For enabling api:
- Open Hydrus Network
- On top menu: Services -> manage services
- Double click *client api*
- enable *run the client api?*
- note the port number of *local port* (by default it is 45869)
- Apply

For creating api key:
- Open Hydrus Network
- On top menu: Services -> review services
- local -> client api
- Click *add* select *manually*
- Enable *permits everything*
- On top you will see your api key (access key)
- Click apply

For creating ai tag domains (optional but highly recommended to keep things clean and organized):
- Open Hydrus Network
- On top menu: Services -> manage services
- Click *add* select *local tag domain*
- Create a local tag domain called *ai tags*
- Create another local tag domain called *ai faces*
- Click apply

9. **You are done setting up and ready to run the scripts. Proceed to the How to Run section**

## How to Run (Basics)

For running the scripts you have to make sure you navigate into the project folder and activate the venv with :
```bash
cd hydrus-ai-taggers
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows
```
#### Quick Test to Check if AI models Works (Optional)
Before you start a tagging job, you can use a command like this to check if you can load the AI model properly and tag it.

`python wd14/main.py evaluate "/home/user/Pictures/my-amazing-image.jpg" --device CPU`

This command will output what tags it would generate for that image.

You can use a command like this to check if you can load the AI model properly and detect faces with it.

`python face/face_main.py evaluate "/home/user/Pictures/my-amazing-image.jpg" --device CPU`

This command will tell you how many faces are detected in that image with its confidence score and face coordinates.

If these work fine, you can successfully load and run AI models.

#### AI Tagging
Start a full-auto AI tagging with a command like:

`python wd14/wd14_main.py full-auto --token YOUR_ACCESS_KEY --tag-service "ai tags" --model "wd-eva02-large-tagger-v3" --device CPU --limit 100 --threshold 0.75`

Replace *YOUR_ACCESS_KEY* with your access key.

Replace *--device CPU* with which hardware you want to use. Options: *CPU, GPU, NPU*. If you do CPU it will be slow.

You can lower threshold to 0.60 or 0.50 if it becomes too strict. But lower threshold will increase risk of false positives.

When it starts running, it will start generating tags in *ai tags* and files that have *wd14 ai tagged* tag will be skipped.

When you want to stop the process, press *CTRL+C* to kill the process safely.

#### Face Detection and Recognition
Face detection and recognition is a two step process and more complicated than tagging.

First, you have to  start a full-auto AI face detection with a command like this:

`python face/face_main.py auto-detect --token YOUR_ACCESS_KEY --tag-service "ai faces" --device CPU --limit 100 --full-auto`

Replace *YOUR_ACCESS_KEY* with your access key.

Replace *--device CPU* with which hardware you want to use. Options: *CPU, GPU, NPU, OPENVINO*. If you do CPU it will be slow.

Face detection will automatically create a DB containing embeddings data. 

When you want to stop the process, press *CTRL+C* to kill the process safely.

After you have detected some faces and added them into DB, use a command like this to start clustering faces, recognizing people and creating person tags:

`python face/face_main.py recognize-staged --token YOUR_ACCESS_KEY --tag-service "ai faces" --max-distance 0.5 --distance-method cosine_similarity --stages 20,5,3,1`

This uses a staged calculation for lowering risk of false positives and recognizes people better. It first becomes picky about creating new persons, requiring high amount of similar faces. After creating base persons, it gradually runs with less strict filter and assigns them to created persons first, only creating them as new people if no similar face group exists. This gives highly accurate and reliable results.

When this process is over, if your DB had enough detected faces of same people, you will see that it has created person tags like *person:p1*, *person:p2* etc in *ai faces*.

If you know who *person:p1* is, create a tag sibling to it and turn it into desired tag such as *person:bob marley*

If an image is detected and added into DB but not tagged after recognize step, it means there's not enough similar images in DB yet. As you detect and add more images into your DB, recognition step will be more accurate (but will take longer as well).

I suggest you to backup your face_embeddings.db regularly

Face Detection and Recognition works best on real life photography and may not give optimal results on drawn artworks or AI generated images.
