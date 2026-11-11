---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - ultralytics
  - yolo
  - yolo26
  - object-detection
  - building-defects
  - crack-detection
  - building-inspection
  - built-environment
  - iaac-maai
  - ai-methods-1
datasets:
  - senthilsk/crack_detection_dataset
metrics:
  - map50
  - map
---

# Building defect detector (YOLO26n): cracks, stair-step cracks, mold, peeling paint and water seepage

This model finds **building defects in photos and videos of walls and ceilings**: cracks, stair-step cracks in masonry, mold, peeling paint and water seepage. It draws a box with a class and a confidence around every defect it finds. It is meant to **pre-screen** condition-survey photos and walk-through videos for a building surveyor or facility manager, so that the images to check first come to the top. It is a **YOLO26n** detector (2.5 million parameters), pretrained by Ultralytics on COCO and fine-tuned on 2,060 labelled defect images. In our pipeline it is combined with a crack-line filter and **MobileSAM** to turn its boxes into segmentation masks (see *On video*).

It was built as a course assignment for **AI Methods I** (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/group1_day4_assignment
- **Team:** Amelia, Jerome, Kazuko, Sahil (team 1)

## Model at a glance

![Test images: true boxes (top) and predictions (bottom)](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/predictions_vs_truth.png)

*Test images (never seen in training): true boxes on top, predictions below (confidence ≥ 0.15). Cracks and stair-step cracks are found exactly; mold and peeling paint are found, but split into more boxes than the labels; in the room on the right, the 10 water-seepage boxes are missed and only one patch of mold is found.*

![Per-class average precision on the test set](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/per_class_ap.png)

*Line defects are reliable (crack AP50 0.93, stair-step crack 0.77); surface defects are not (mold 0.29, water seepage 0.28, peeling paint 0.05).*

| Model (test set, 446 images) | Crack recall | Crack precision | Recall, all classes | mAP50 |
|---|---|---|---|---|
| Baseline: pretrained YOLO26s (COCO, no fine-tuning) | 0.00 | 0.00 | 0.00 | 0.00 |
| Baseline: rule-based crack detector (OpenCV black-hat + Otsu) | 0.45 | 0.41 | 0.34 | – |
| **This model** (confidence 0.15) | **0.92** | **0.71** | **0.81** | **0.46** |

## Intended use

- **Intended users and use:** building surveyors, facade inspectors, facility managers and students who want to **pre-screen** a set of photos or a video of walls and ceilings: which images show a crack, mold, peeling paint or water damage, and which to look at first. Best suited to **cracks** in plaster, render, concrete and masonry.
- **Out-of-scope uses:**
  - Any **structural safety assessment**: the model says nothing about the width, depth, cause or danger of a crack.
  - Any decision about a property (legal, insurance, sale, rent) **without a qualified person checking every result**. "No detection" means "not found", never "no defect".
  - Counting or measuring individual surface defects (mold, stains, peeling paint): their detection is unreliable (see *Evaluation*).
  - Images very different from the training data: drone or satellite views, thermal images, night images, facades seen from far away, materials such as wood, metal or glass.
  - Monitoring people.

## Model details

| | |
|---|---|
| **Task** | Object detection (boxes, classes and confidences) |
| **Architecture** | Ultralytics YOLO26n: 2.5 million parameters (2.4 M after layer fusion), 5.4 GFLOPs |
| **Base model** | `yolo26n.pt`, pretrained on COCO (80 everyday classes, none of them a defect) |
| **Input** | An RGB image of any size, resized to 640 × 640 pixels internally |
| **Output** | For each defect: a box (x1, y1, x2, y2 in pixels), one of the 5 classes below and a confidence between 0 and 1 |
| **Files** | `model.pt` (weights for Ultralytics, 5.5 MB); `results.csv` (training log, one row per epoch); `experiments.csv` (the three runs compared on validation); `video_*.mp4` (three annotated videos); the figures of this card |
| **Speed** | About 2 ms per image on a laptop RTX 5080 GPU; about 30 ms per image on a recent high-end laptop CPU (AMD Ryzen AI 9 HX 375). Older or lighter CPUs will be slower, but the nano model stays usable without a GPU. |
| **Library versions** | ultralytics 8.4.190, torch 2.14.1, Python 3.12 |

### Classes

The dataset comes without written rules; these definitions are ours, based on how the boxes are drawn.

| ID | Class | What counts | What does not |
|---|---|---|---|
| 0 | `crack` | A visible fracture line in plaster, paint, render or concrete, any direction or length | Joints between tiles or panels, shadows, drawn lines |
| 1 | `mold` | Dark (black, green, grey) spotty growth on walls, ceilings or window frames | Uniform dirt or water stains without spots (`water_seepage`) |
| 2 | `peeling_paint` | Paint or plaster that has lifted, flaked or fallen off, exposing the layer below | Intact paint of different colours |
| 3 | `stairstep_crack` | A crack in masonry (brick, block) that follows the mortar joints in a step pattern, a sign of settlement | Straight cracks through a plastered wall (`crack`) |
| 4 | `water_seepage` | Stains, tide marks or discoloured patches caused by water coming through a wall or ceiling | Mold spots growing on the stain (`mold`) |

The classes overlap in reality: water causes stains, then mold and peeling paint, often in the same place.

## Training data

| | |
|---|---|
| **Dataset** | *detr_crack_dataset*, mirrored on Hugging Face as [`senthilsk/crack_detection_dataset`](https://huggingface.co/datasets/senthilsk/crack_detection_dataset) (revision `03ae44b`) |
| **Source** | [Roboflow Universe](https://universe.roboflow.com/objectdetection-qxiqx/detr_crack_dataset) |
| **Licence** | CC BY 4.0 |
| **Size** | 2,950 images and 3,708 boxes: `crack` 2,805, `stairstep_crack` 311, `peeling_paint` 309, `water_seepage` 149, `mold` 134 |
| **Splits** | Our own split, 70 / 15 / 15 %: 2,060 training, 444 validation, 446 test images. Split **by source** (images with the same original file name or a near-identical 8 × 8 hash stay together) and stratified by the rarest class in each image. The original split was not used: its test set had no `stairstep_crack`. |
| **Coverage** | About 80 % of the images are 200 × 200 px close-ups of a single crack on a plain painted or concrete surface; the other classes come from a few hundred larger web photos of interior walls and ceilings and of brick walls, some with watermarks of stock-photo and inspection websites. Almost every image contains a defect. |
| **Privacy** | The YuNet face detector found a face in 6 of the 2,950 images; faces are blurred in every figure and video frame we publish. |

![Examples of each class](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/dataset_examples.png)

*One row per class, with the labelled boxes. Crack boxes are loose (a thin line in a mostly intact wall); surface defects are often labelled with many small boxes.*

![Boxes per class and split](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/class_counts.png)

*Very imbalanced: three quarters of the boxes are `crack`; `mold` and `water_seepage` have about 20 boxes each in the test set.*

## Training procedure

- **Fine-tuning** of `yolo26n.pt` for **60 epochs**, image size 640, batch 32, early-stopping patience 20, Ultralytics default optimiser and learning-rate schedule, `seed=42`, deterministic.
- **Augmentation** (Ultralytics defaults): mosaic (switched off for the last 10 epochs), random scaling, translation, horizontal flips and colour changes.
- **Hardware and time:** one NVIDIA RTX 5080 Laptop GPU (16 GB), **11.7 minutes**.
- **Model selection:** three runs, each changing one thing, compared on the **validation** set; the run with the best validation mAP50-95 was selected, with the weights of its best epoch (`best.pt`). The test set was used **once**, at the end.

| Run | Parameters | Training | Val mAP50 | Val mAP50-95 |
|---|---|---|---|---|
| YOLO26s, 640 px | 10.0 M | 24.5 min | 0.494 | 0.331 |
| **YOLO26n, 640 px (this model)** | **2.5 M** | **11.7 min** | **0.535** | **0.347** |
| YOLO26s, 320 px | 10.0 M | 7.3 min | 0.506 | 0.342 |

With about 2,000 mostly small images, the smaller model is as good as the larger one, and 320 px images as good as 640 px: the limit is the data, not the model.

![Training curves](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/training_curves.png)

*Training curves of the YOLO26s run (the reference run of the experiment; the YOLO26n log is in `results.csv`). The validation mAP50 rises fast thanks to the pretrained weights and levels off at about 0.48–0.50 after epoch 40; the validation losses keep falling with the training losses, so there is no real overfitting.*

## Evaluation

**Overall:**

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Validation (444 images) | 0.535 | 0.347 | 0.633 | 0.526 |
| **Test (446 images, used once)** | **0.464** | **0.297** | **0.524** | **0.456** |

**Per class, on the test set** (precision and recall at Ultralytics' best-F1 confidence):

| Class | Boxes in test set | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| crack | 409 | 0.83 | 0.92 | **0.93** | 0.69 |
| stairstep_crack | 54 | 0.78 | 0.70 | **0.77** | 0.47 |
| mold | 21 | 0.46 | 0.29 | 0.29 | 0.18 |
| water_seepage | 24 | 0.41 | 0.25 | 0.28 | 0.12 |
| peeling_paint | 33 | 0.15 | 0.12 | 0.05 | 0.02 |

The rare classes have only 21–54 test boxes: their scores are uncertain by about ±0.1.

![Confusion matrix on the test set](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/confusion_matrix.png)

*How to read it: each column is a true class and sums to 1; the bottom row ("background") is the share of defects that were **missed**, the right-hand column what the false alarms were. Cracks are found 98 % of the time and stair-step cracks 91 %; 42 % of the peeling paint is missed, and 29 % of the water seepage is called peeling paint. Most false alarms are `crack`, the most frequent class.*

**Where it fails:**

![Failures on the validation set](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/failures.png)

*Top: the images with the most missed defects; bottom: false alarms (green = truth, red = prediction, confidence ≥ 0.25).*

- **Box granularity:** on stains, peeling paint and crack networks the annotators sometimes drew many small boxes, sometimes one large one; the model draws a few large ones. Under the IoU ≥ 0.5 rule, one large box over three small true boxes counts as one false alarm and three misses, although a person would call it correct. The low scores of the surface classes are partly a **labelling** problem.
- **Scattered mold spots** are found as one dense area, not cluster by cluster.
- **Many "false alarms" are duplicates or different splits of a real defect** (two boxes on one stair-step crack); false alarms on intact wall are rarer.
- **Surface classes are confused** with each other (water seepage ↔ peeling paint ↔ mold): they often appear together and look alike.

### Choosing a confidence threshold

| Threshold (test set, all classes) | Precision | Recall |
|---|---|---|
| 0.05 | 0.43 | 0.83 |
| **0.15 (recommended)** | **0.65** | **0.81** |
| 0.30 | 0.79 | 0.78 |
| 0.50 | 0.89 | 0.74 |
| 0.70 | 0.94 | 0.61 |

![Precision and recall against the confidence threshold](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/threshold_sweep.png)

**Recommendation: 0.15.** We chose it on the **validation** set as the threshold with the highest F2 score (recall counts twice as much as precision), because for pre-screening a missed defect costs more than a false alarm; on the test set it finds **81 %** of the defects (437 found, 239 false alarms, 104 missed). Use 0.3 if false alarms matter more. For mold, water seepage and peeling paint no threshold gives a good recall (mold at most 0.43): this needs more and better-labelled data.

### On new photos

![Eight new photos ranked by priority](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/application.png)

*Eight openly licensed photos from Wikimedia Commons (not in the dataset), ranked by the sum of the detection confidences. Damage is found in all eight and the most damaged rooms come first, but the type is often wrong: cracks in masonry are always called `stairstep_crack`, the mold on the ceiling (#4) is called `water_seepage`, and in #7 a pole against the sky is taken for a crack.*

### On video

We ran the model on three openly licensed videos (every second frame, confidence 0.15) and turned each box into a **segmentation mask**: crack lines are traced inside the box with a black-hat filter and Otsu threshold, and surface defects (mold, peeling paint, water seepage) are outlined by **MobileSAM** prompted with the box. SAM alone, given a crack box, outlines the wall next to the crack rather than the crack itself.

**1. Cracked plastered wall:** the crack is found in **100 %** of the frames (mean confidence 0.80) and traced exactly.

<video controls width="100%" src="https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/video_crack_white_wall.mp4"></video>

**2. Brick wall with a stepped crack:** recognised as `stair-step crack` in only **10 %** of the frames, with a low confidence: a hole with missing bricks is not in the training data.

<video controls width="100%" src="https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/video_brick_wall_step_crack.mp4"></video>

**3. Derelict apartment (a false-alarm test):** hardly any wall defects, yet low-confidence alarms (mean 0.36): the water in a mop bucket is taken for `peeling paint`, stains and shadows for `water seepage`. A clean mask is not a sign that a detection is correct.

<video controls width="100%" src="https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/video_derelict_apartment.mp4"></video>

![Three frames per video and the segmented area over time](https://huggingface.co/iaac-maai/day4-team1-building-defect-detector/resolve/main/video_frames.png)

*Three frames per video and the segmented defect area (% of the frame) over time.*

*Videos: "A crack on the wall" by Monstera Production and "Broken brick wall" by Kmeel.com Videos ([Pexels licence](https://www.pexels.com/license/)); "Water Dripping into a Bucket in a Derelict Apartment in Canada" by Gogerr ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Water_Dripping_into_a_Bucket_in_a_Derelict_Apartment_in_Canada.webm), CC BY 4.0; our annotated version is shared under the same licence). Faces are blurred automatically before a frame is written.*

## Limitations and biases

- **Surface defects are unreliable** (AP50 0.05–0.29): few training examples (87–247 boxes), overlapping definitions and inconsistent box sizes in the labels.
- **Training data:** about 80 % close-ups of single cracks on plain walls; few facades, few materials beyond plaster, concrete and brick; mostly web photos. Almost every image contains a defect, so the model has seen little intact wall and raises false alarms on unfamiliar scenes (mop buckets, hedges, poles against the sky).
- **Masonry cracks** are almost always called `stairstep_crack`, whatever their shape.
- **Small test sets** for the rare classes (±0.1 uncertainty). The split by source relies on file names and an image hash, so very similar crops of one photo could still sit in different splits, which would make the scores slightly optimistic.
- **Provenance:** the dataset is published under CC BY 4.0 by its uploader, but some images carry watermarks of stock-photo and inspection websites.

## Ethics and privacy

The training images show buildings, not people; the YuNet face detector found a face in 6 of 2,950 images, and every figure and video frame published here is blurred where a face is detected. If you use the model on your own survey photos or videos, **blur faces and number plates** before sharing them, and get the permission of the occupants for interiors. The model must not be used to monitor people or to make decisions about the occupants of a building.

## How to use

```python
# pip install ultralytics==8.4.190 huggingface_hub
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("iaac-maai/day4-team1-building-defect-detector", "model.pt"))

result = model.predict("wall_photo.jpg", conf=0.15)[0]          # 0.15 = our recommended threshold (favours recall)
for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
    print(model.names[int(cls)], f"{float(conf):.2f}", [round(v) for v in box.tolist()])
result.save("wall_photo_defects.jpg")                           # the photo with the boxes drawn
```

The class names are stored in the weights (`model.names`: `crack`, `mold`, `peeling_paint`, `stairstep_crack`, `water_seepage`). On a video, call `model.predict("video.mp4", conf=0.15, stream=True)` and loop over the results. The segmentation step (crack-line filter and MobileSAM) is in section 7.2 of the notebook.

⚠️ `.pt` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above.

## Reproducibility

The full pipeline (download, split, baselines, training, evaluation, threshold, photo report, video segmentation) is in the GitHub repository: https://github.com/iaac-maai/group1_day4_assignment. Run `notebook/object_detection.ipynb` with the packages in `requirements.txt`; without a GPU it downloads these weights instead of training.

## Authors and licence

Amelia, Jerome, Kazuko, Sahil (team 1). Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

- **Model:** released under **AGPL-3.0**, the licence of the Ultralytics YOLO models it is fine-tuned from. If you build a product or service on it, AGPL-3.0 requires you to share your source code, or you need an Ultralytics Enterprise licence.
- **Training data:** *detr_crack_dataset*, Roboflow Universe, CC BY 4.0.
- **Photos and videos in this card:** the application photos come from Wikimedia Commons (CC BY, CC BY-SA, CC0 and public domain; authors listed in the notebook, section 7.1); the videos are credited under *On video*.
