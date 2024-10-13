import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image, make_grid
from pathlib import Path
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from grm import GenerativeRepresentation
from tqdm import tqdm

from configs.default_cifar10_configs import get_config
from torchmetrics.image.fid import FrechetInceptionDistance
from consistency_models.consistency_models import improved_timesteps_schedule

# Argument parser
parser = argparse.ArgumentParser(description='grm')
parser.add_argument('--beta', default=0.3, type=float, help='loss weight of contrastive loss')
parser.add_argument('--data', default='/local_datasets/miniimagenet', type=Path, metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--img-size', default=32, type=int, help='size of input data')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--output-dir', default='./output/', type=Path, metavar='DIR', help='path to output directory')  
parser.add_argument('--resume', type=str, default=None, help='path to resume checkpoint')

args = parser.parse_args()

def main():
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/checkpoints').mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    
     # Training configurations
    batch_size = args.batch_size
    num_epochs = args.epochs

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    fid_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    fid_loader = torch.utils.data.DataLoader(dataset=fid_dataset, batch_size=batch_size, shuffle=False)


    total_training_steps = len(train_loader)
    total_training_steps = num_epochs * len(train_loader)

    model_config = get_config()
    model = GenerativeRepresentation(
        model_config = model_config,
        total_step = total_training_steps,
        ema_decay = 0.99993,
        rep_dim = 256,
        emb_dim = 1024,
        batch_size = batch_size,
        num_classes = 10,
        reg_weight = 1.0,
        lambd = 0.0051,
        is_stochastic = False
    ).to(device)

    # Load checkpoint if provided
    start_epoch, current_training_step = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        current_training_step = (checkpoint['epoch'] + 1) * len(train_loader)
        print(f"Resuming from epoch {start_epoch}")

    print("\n--- Training Configuration ---")
    print(f"Number of Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"Output Directory: {args.output_dir}\n")

    print("--- Model Summary ---")
    model_params = sum(p.numel() for p in model.parameters()) / 1_000_000  # Convert to millions
    print(f"Model Parameters: {model_params:.2f} M")
    print("-----------------------\n")
    

    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=1000)

    fid = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(device)

    for _, batch in tqdm(enumerate(fid_loader), total=len(fid_loader), desc="Calculating FID"):
        fid.update(batch[0].to(device), real=True)
    
    for epoch in range(start_epoch, num_epochs):
        for _, (images, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):

            images = images.to(device)
            optimizer.zero_grad()
            recon_loss, rep_loss = model(images, current_training_step)
            loss = recon_loss + args.beta * rep_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.update_ema()
            current_training_step += 1

            # Logging to TensorBoard
            if (current_training_step) % 10 == 0:
                writer.add_scalar('Loss/recon_loss', recon_loss.item(), current_training_step)
                writer.add_scalar('Loss/rep_loss', rep_loss.item(), current_training_step)
                writer.add_scalar('Training/epoch', epoch + 1, current_training_step)
                writer.add_scalar('num_timestep', improved_timesteps_schedule(current_training_step, total_training_steps), current_training_step)
                
        # Sample and log images
        
        samples_one_step_ema = (model.get_sample(sigmas=(80, ), num_sample=8, ema=True) / 2 + 0.5).clamp(0, 1).cpu().detach()
        samples_one_step_ema = make_grid(samples_one_step_ema)
        writer.add_image('Sample/one_step_ema', samples_one_step_ema, epoch + 1)

        samples_few_step_ema = (model.get_sample(sigmas=(80.0, 24.4, 5.84, 0.9, 0.661), num_sample=8, ema=True) / 2 + 0.5).clamp(0, 1).cpu().detach()
        samples_few_step_ema = make_grid(samples_few_step_ema)
        writer.add_image('Sample/few_step_ema', samples_few_step_ema, epoch + 1)

        batch = (images[:8] / 2 + 0.5).clamp(0, 1).cpu().detach()
        batch = make_grid(batch)
        writer.add_image('Sample/batch', batch, epoch + 1)

        aug1, aug2 = model.get_augmentation_sample(images, current_training_step, num_samples=8)
        aug1 = (aug1 / 2 + 0.5).clamp(0, 1).cpu().detach()
        aug2 = (aug2 / 2 + 0.5).clamp(0, 1).cpu().detach()
        aug1 = make_grid(aug1)
        aug2 = make_grid(aug2)
        writer.add_image('Sample/augmentation1', aug1, epoch + 1)
        writer.add_image('Sample/augmentation2', aug2, epoch + 1)


        if (epoch + 1) % 10 == 0: 
            for _ in range(int(50000 / batch_size)):
                with torch.no_grad():
                    samples = model.get_sample(sigmas=(80.0, ), num_sample = batch_size, ema=True)
                    image = (samples / 2 + 0.5).clamp(0, 1)
                    fid.update(image, real=False)
            fid_result = float(fid.compute())
            print(f"Epoch [{epoch+1}], FID: {fid_result}")
            writer.add_scalar('FID', fid_result, epoch + 1)
            fid.reset()
            torch.cuda.empty_cache()

        if (epoch + 1) % 20 == 0: 
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),                
            }
            torch.save(checkpoint, f'{args.output_dir}/checkpoints/grm_{epoch}.pth')


    writer.close()

if __name__ == '__main__':
    main()


