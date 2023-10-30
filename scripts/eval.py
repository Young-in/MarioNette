"""Train script."""
import os
import logging

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import ttools

from marionet import datasets, models, callbacks
from marionet.interfaces import Interface
from PIL import Image

LOG = logging.getLogger(__name__)

th.backends.cudnn.deterministic = True


def _worker_init_fn(_):
    np.random.seed()


def _set_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGBA', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def main(args):
    # LOG.info(f"Using seed {args.seed}.")

    device = "cuda" if th.cuda.is_available() and args.cuda else "cpu"

    learned_dict = models.Dictionary(args.num_classes,
                                     (args.canvas_size // args.layer_size*2,
                                      args.canvas_size // args.layer_size*2),
                                     4, bottleneck_size=args.dim_z)
    learned_dict.to(device)

    model = models.Model(learned_dict, args.layer_size, args.num_layers)
    model.eval()

    model_checkpointer = ttools.Checkpointer(
        os.path.join(args.checkpoint_dir, "model"), model)
    model_checkpointer.load_latest()

    learned_dict, dict_codes = model.learned_dict()
    print(f"model:{model}")
    print(f"model_summary:")
    summary(model)
    print(f"learned_dict:{learned_dict.shape}")
    print(f"dict_codes:{dict_codes.shape}")

    imgs = []

    for idx in range(learned_dict.shape[0]):
        t = (learned_dict[idx].permute(1, 2, 0).detach().cpu().numpy() * 256).astype(np.uint8)
        img = Image.fromarray(t, mode = "RGBA")
        img.save(f"/home/youngin/MarioNette/result/image_{idx}.png")

        imgs.append(img)
    
    for group in range(len(imgs) // 15):
        image_grid(imgs[group * 15:(group + 1) * 15], 3, 5).save(f"/home/youngin/MarioNette/result_grid_{group}.png")



if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()

    # Representation
    parser.add_argument("--layer_size", type=int, default=8,
                        help="size of anchor grid")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of layers")
    parser.add_argument("--num_classes", type=int, default=150,
                        help="size of dictioanry")
    parser.add_argument("--canvas_size", type=int, default=128,
                        help="spatial size of the canvas")
    
    parser.add_argument("--dim_z", type=int, default=128)

    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
