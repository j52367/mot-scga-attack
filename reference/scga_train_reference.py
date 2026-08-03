import copy
import datetime
import os
import shutil
import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from tqdm import tqdm

import generator
import surrogate
from datasets.paths import DATA_ROOT_DIR
from datasets.train import train_dataset_selection
from generator import build_generator
from generator.resnet import weights_init_normal
from utee.misc import AverageMeter
from utee.parser import get_parser

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Parse arguments and set up
args = get_parser().parse_args()
seed_everything(args.seed)
device = torch.device(
    f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
)

# Set up output directory
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_T%H%M%S")[2:]
args.output_dir = '{}/{}_{}_{}_{}'.format(
    args.output_dir, now, args.method, args.generator, args.surrogate
)
os.makedirs(args.output_dir, exist_ok=True, mode=0o777)
log_file = os.path.join(args.output_dir, 'training.log')

# Copy and save the current file to output_dir
current_file_path = __file__
output_file_path = os.path.join(
    args.output_dir, os.path.basename(__file__)
)
shutil.copy(current_file_path, output_file_path)

# Initialize surrogate model
if args.surrogate == 'vgg16':
    model = surrogate.Vgg16()
    layer_idx = 16  # Maxpooling.3
    # conv4-1 # 18 for LTP
elif args.surrogate == 'vgg19':
    model = surrogate.Vgg19()
    layer_idx = 18  # Maxpooling.3
elif args.surrogate == 'res152':
    model = surrogate.Resnet152()
    layer_idx = 5  # Conv3_8
elif args.surrogate == 'dense169':
    model = surrogate.Dense169()
    layer_idx = 6  # Denseblock.2
else:
    raise ValueError('Please check the surrogate type')

if args.layer_idx is not None:
    layer_idx = args.layer_idx

print(vars(args))

model = model.cuda()
model.eval()

# Input dimensions
scale_size = 256
img_size = 224


def feature_map_similarity_hinge(
    fmap_ref: torch.Tensor,
    fmap_tgt: torch.Tensor,
    threshold: float = 0.6
) -> torch.Tensor:
    """Compute hinge loss based on feature map cosine similarity.

    Args:
        fmap_ref: Reference feature map tensor of shape (B, C, H, W).
        fmap_tgt: Target feature map tensor of shape (B, C, H, W).
        threshold: Threshold for hinge loss. Defaults to 0.6.

    Returns:
        Mean hinge loss value.
    """
    B = fmap_ref.shape[0]
    # Flatten (B, C*H*W)
    ref_flat = fmap_ref.view(B, -1)
    tgt_flat = fmap_tgt.view(B, -1)
    # Normalize
    ref_n = F.normalize(ref_flat, p=2, dim=1)
    tgt_n = F.normalize(tgt_flat, p=2, dim=1)
    # Cosine similarity
    cos_sim = (ref_n * tgt_n).sum(dim=1)
    # Hinge loss
    return F.relu(threshold - cos_sim).mean()


def update_ema(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    momentum: float = 0.999
) -> None:
    """Update teacher model parameters using exponential moving average.

    Args:
        student: Student model to copy parameters from.
        teacher: Teacher model to update.
        momentum: EMA momentum factor. Defaults to 0.999.
    """
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def normalize(
    t: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> torch.Tensor:
    """Normalize tensor using ImageNet statistics.

    Args:
        t: Input tensor of shape (B, C, H, W).
        mean: Mean values for each channel. Defaults to ImageNet mean.
        std: Standard deviation values for each channel. Defaults to ImageNet std.

    Returns:
        Normalized tensor.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


# Initialize generator (student and teacher)
student = build_generator(
    generator=args.generator,
    gap=args.gap,
    inception=False,
    nat=args.nat
).cuda()
student = torch.compile(student)
teacher = copy.deepcopy(student)
for p in teacher.parameters():
    p.requires_grad = False
student.apply(weights_init_normal)
teacher.apply(weights_init_normal)
optimG = optim.Adam(student.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Data transforms
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

# Load training data
print('loading training data...')
g = torch.Generator()
g.manual_seed(0)

train_set = train_dataset_selection(
    dataset_name=args.train_data,
    root_dir=DATA_ROOT_DIR,
    train_transform=data_transform
)[0]
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    generator=g,
)
train_size = len(train_set)
print(f'Training data: {args.train_data}, data size: {train_size}')
args.output_dir = '{}/{}_{}_{}'.format(
    args.output_dir, args.method, args.generator, args.surrogate
)
os.makedirs(args.output_dir, exist_ok=True)

# Initialize loss tracking
loss_meter = AverageMeter()
loss_dict = {}
eps = args.eps / 255.0

for epoch in range(args.epochs):
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{args.epochs}",
        total=len(train_loader),
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                   '[{elapsed}<{remaining}]',
        ncols=50
    )
    for i, (img, label) in enumerate(pbar):
        img = img.cuda()
        student.train()
        optimG.zero_grad()

        # 1) Teacher forward (clean path)
        with torch.no_grad():
            _, feats_teacher = teacher(img, feat=True)

        # 2) Student forward (adversarial)
        adv, feats_student = student(img, feat=True)

        # Unbounded
        adv_unbounded = adv.clone()

        # Projected (bounded): within eps and [0, 1]
        adv = torch.clamp(
            torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0
        )

        # 3) Similarity loss
        taus = [0.6 + 0.05 * i for i in range(6)]  # thresholds for 6 blocks
        gen_feature_sim_loss = 0.0
        for j in range(len(feats_teacher[1:]) - 4):
            gen_feature_sim_loss += feature_map_similarity_hinge(
                feats_teacher[1:][j],
                feats_student[1:][j],
                threshold=taus[j]
            )

        # 4) Cosine similarity loss on surrogate
        img_out_slices = model(normalize(img.clone()))
        adv_out_slices = model(normalize(adv.clone()))
        img_out_slice = img_out_slices[layer_idx]
        adv_out_slice = adv_out_slices[layer_idx]

        # Attention: filler for later integration with DA or RN module
        # of BIA method
        attention = torch.ones(adv_out_slice.shape).cuda()
        adv_out_flat = (adv_out_slice * attention).reshape(
            adv_out_slice.shape[0], -1
        )
        img_out_flat = (img_out_slice * attention).reshape(
            img_out_slice.shape[0], -1
        )
        loss_BIA = torch.cosine_similarity(
            adv_out_flat, img_out_flat
        ).mean()

        loss = loss_BIA + 0.7 * gen_feature_sim_loss
        loss_dict['BIA'] = loss_BIA.item()
        loss_dict['GenFeat'] = gen_feature_sim_loss.item()
        loss.backward()
        optimG.step()

        update_ema(student, teacher)

        loss_meter.update(loss.item(), img.size(0))

        if i > 0 and i % args.save_freq == 0:
            torch.save(
                teacher.state_dict(),
                os.path.join(
                    args.output_dir,
                    'teacher_{}_{}.pth'.format(args.method, epoch)
                )
            )
