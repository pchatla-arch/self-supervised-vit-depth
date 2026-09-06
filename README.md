# Self-Supervised Monocular Depth Estimation Using Vision Transformers

**Prasuna Chatla**

*Robotics Institute, Carnegie Mellon University*

A PyTorch implementation of self-supervised monocular depth estimation using a Vision Transformer (ViT) encoder, a multi-scale CNN decoder, relative-pose estimation, differentiable view reconstruction, SSIM/L1 photometric loss, and edge-aware disparity smoothness.

The trained depth network predicts a dense depth map from a **single RGB image**. During training, adjacent video frames and camera intrinsics provide the geometric signal, so ground-truth depth is not required.

## Highlights

- Vision Transformer encoder for global scene context
- Multi-scale CNN decoder for dense depth prediction
- Pose network for target-to-source camera transformation
- Differentiable back-projection, projection, and image warping
- SSIM + L1 photometric reconstruction objective
- Edge-aware disparity smoothness
- CNN baseline and standard depth-evaluation metrics
- Training-loss, qualitative-depth, resolution, initialization, and post-processing experiments

## Conceptual Overview

![Conceptual overview of self-supervised monocular depth estimation](assets/conceptual_overview.png)

The upper path shows self-supervised training from target and source frames. The lower path shows monocular inference from one RGB image.

## Self-Supervised Learning Framework

![Self-supervised depth learning framework](assets/self_supervised_learning_framework.png)

For target image $I_t$, the depth network predicts $D_t$. The pose network estimates $T_{t\rightarrow s}$ between the target and a source frame $I_s$. Predicted depth, relative pose, and camera intrinsics $K$ reconstruct the target image through differentiable warping.

The training objective is:

$$
L_{\mathrm{photo}} = \alpha\frac{1-\operatorname{SSIM}(I_t,\hat{I}_t)}{2}
+ (1-\alpha)\left\|I_t-\hat{I}_t\right\|_1,
$$

$$
L_{\mathrm{smooth}} = |\partial_x d|e^{-|\partial_x I|}
+ |\partial_y d|e^{-|\partial_y I|},
$$

$$
L_{\mathrm{total}} = \sum_s w_s\left(L_{\mathrm{photo}}^{(s)}
+ \lambda_sL_{\mathrm{smooth}}^{(s)}\right).
$$

## Model Architecture

### ViT Encoder

The input image is divided into non-overlapping $16 \times 16$ patches. The custom transformer configuration uses:

| Parameter | Value |
| --- | ---: |
| Patch size | 16 x 16 |
| Embedding dimension | 768 |
| Attention heads | 8 |
| Transformer blocks | 6 |
| MLP hidden dimension | 3072 |

### Multi-Scale CNN Decoder

The decoder progressively restores spatial resolution and predicts depth at full, half, and quarter scales. A representative channel progression is:

```text
768 -> 256 -> 128 -> 64 -> 32 -> 1
```

The experimental scale weights are:

```text
[1.0, 0.5, 0.25]
```

## Repository Layout

```text
models/
  vit_encoder.py       # patch embedding and transformer blocks
  depth_decoder.py     # multi-scale convolutional decoder
  depth_model.py       # end-to-end depth model
  pose_net.py          # relative camera pose network
  cnn_baseline.py      # convolutional baseline
losses/
  ssim.py
  photometric.py
  smoothness.py
  self_supervised.py
geometry/
  projection.py        # depth conversion, SE(3), projection, and warping
datasets/
  triplet_dataset.py   # target/source frame manifest loader
utils/
  metrics.py           # AbsRel, SqRel, RMSE, RMSE-log, delta metrics
scripts/
  train.py
  evaluate.py
  predict.py
configs/
  kitti.yaml
docs/
  PROJECT_REPORT.md
  Self-Supervised-Monocular-Depth-Estimation-Using-Vision-Transformers.pdf
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Training Data Format

Self-supervised training requires temporally adjacent frames from the same calibrated camera sequence.

```csv
prev,target,next,fx,fy,cx,cy
sequence/000000.png,sequence/000001.png,sequence/000002.png,721.5,721.5,609.6,172.9
```

Camera intrinsics must correspond to the resized images.

## Train

```bash
python scripts/train.py \
  --manifest data/train_triplets.csv \
  --root /path/to/dataset \
  --epochs 20 \
  --batch-size 4 \
  --height 192 \
  --width 640
```

Checkpoints are written to `checkpoints/last.pt` by default.

## Predict Depth from One Image

```bash
python scripts/predict.py \
  --image example.jpg \
  --checkpoint checkpoints/last.pt \
  --output depth.png
```

Only one RGB image is required at inference time.

## Evaluate

The evaluator expects a CSV containing `image,depth` columns. Ground-truth depth is used only for evaluation.

```bash
python scripts/evaluate.py \
  --manifest data/eval.csv \
  --checkpoint checkpoints/last.pt
```

Reported metrics:

- Absolute Relative Error (`AbsRel`)
- Squared Relative Error (`SqRel`)
- RMSE
- RMSE-log
- $\delta < 1.25$
- $\delta < 1.25^2$
- $\delta < 1.25^3$

Median scaling is supported because monocular self-supervised depth has global-scale ambiguity.

## Experiments and Results

### Experimental Setup

The recorded experiments used a custom ViT/CNN depth estimator, multi-scale predictions, Adam-based optimization, and training runs extending to 500 epochs. Input sizes between 224 x 224 and 256 x 256 were explored for the image-level experiments; the temporal training script defaults to 192 x 640 for driving sequences. The experiments were conducted in Google Colab Pro with an NVIDIA CUDA-capable GPU, with approximately 16 GB of memory required for larger configurations.

| Experiment | Main observation |
| --- | --- |
| Training convergence | Several runs showed a rapid early loss decrease followed by a low plateau; convergence speed and late-stage noise varied |
| Input resolution | Higher-resolution images retained more spatial detail; lower-resolution runs were less stable and sometimes required longer training |
| Transformer initialization | Pretrained ViT initialization converged faster and produced stronger qualitative outputs than random initialization |
| Composite loss | Photometric reconstruction, SSIM, and edge-aware smoothness together produced the most stable behavior |
| Smoothness weight | Excessive weighting introduced artifacts or softened fine boundaries |
| Multi-scale prediction | Coarse outputs preserved global layout while fine outputs retained more local structure |
| Bilateral filtering | Filtering reduced local variation and produced smoother maps, with a trade-off in fine boundary detail |
| Scene diversity | Depth maps retained broad foreground/background organization across animal, landscape, urban, crowd, indoor, repetitive-pattern, and underwater scenes |

### Training-Loss Behavior

![Training-loss curves from four experiments](assets/training_loss_comparison.png)

Four runs show a strong initial decrease in loss. Three approach a low plateau, while one remains more variable.

![Training loss over 500 epochs](assets/training_loss_500_epochs.png)

The 500-epoch run also converges sharply during the early phase and shows intermittent late-stage spikes.

### Qualitative Depth Predictions

![Representative deer and city depth predictions](assets/qualitative_depth_examples_deer_city.png)

![Representative street, skyline, and animal depth predictions](assets/qualitative_depth_examples_urban_animals.png)

The examples preserve broad scene organization, including foreground/background separation, sky/ground structure, and dominant object boundaries. Fine details remain challenging in crowded scenes, repetitive textures, thin structures, and low-texture regions.

### Post-Processing

![Raw and bilateral-filtered depth comparisons](assets/postprocessing_comparison.png)

Bilateral filtering generates smoother visual output. Raw and filtered predictions are kept separate because post-processing can soften fine depth discontinuities.

### 500-Epoch Multi-View Result

![500-epoch multi-view depth result](assets/multiview_500_epoch_depth_result.png)

The recorded multi-view experiment preserves the dominant animal and branch structure; bilateral filtering produces a smoother version of the prediction.

### Result Scope

This release reports training-loss behavior and qualitative depth maps. A fixed benchmark split, complete per-run metadata, and standard metric values were not captured for these experiment runs, so the repository does not claim benchmark depth accuracy.

The complete experiment discussion and extended result galleries are available in the [project report](docs/PROJECT_REPORT.md).

## Tests

```bash
pytest -q
```

The tests cover model output shapes and the differentiable geometry path.

## Limitations

- Monocular depth has inherent global-scale ambiguity.
- Dynamic objects, occlusions, reflections, and low-texture regions can violate photometric assumptions.
- Larger transformer configurations require substantial GPU memory.
- Dataset-specific cropping, intrinsic rescaling, dynamic-object handling, and reproducible splits are required for benchmark evaluation.

## Future Work

- Evaluate on a fixed KITTI split with standard metrics
- Add minimum reprojection, auto-masking, occlusion handling, and dynamic-object masking
- Compare CNN, ViT, and Swin Transformer encoders
- Add matched pretrained-versus-random initialization experiments
- Measure peak memory, latency, and throughput at multiple resolutions
- Explore pruning, quantization, knowledge distillation, and lightweight transformer variants

## Project Report

- [Detailed Markdown report](docs/PROJECT_REPORT.md)
- [Formatted PDF report](docs/Self-Supervised-Monocular-Depth-Estimation-Using-Vision-Transformers.pdf)
- [Editable Word report](docs/Self-Supervised-Monocular-Depth-Estimation-Using-Vision-Transformers.docx)
