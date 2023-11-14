#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

import argparse
import os
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import env as reasoning_env
from util.averagemeter import AverageMeter
from util.checkpath import check_paths, save_checkpoint
from backend_dataset import RAVENDataset
from AttributeLevelReasoner import AttributeLevelReasoner
import losses
from raven_one_hot import create_one_hot
from VSAConverter import generate_nvsa_codebooks

SceneProb = namedtuple("SceneProb", ["position_prob", "number_prob", "type_prob", "size_prob", "color_prob"])

########### Gaussian smoothing to simulate more unceratinty ###############################
def gaussian_kernel(size: int, mean: float, std: float):
    """Generate a Gaussian kernel."""
    x = torch.linspace(-mean, mean, steps=size)
    kernel = torch.exp(-x**2 / (2*std**2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_smooth(tensor, std, random_range=0.1, kernel_size=5):
    randomized_std = torch.abs(torch.normal(mean=std, std=random_range, size=(1,))).item()
    randomized_std = max(1e-5, randomized_std)
    kernel = gaussian_kernel(kernel_size, 0, randomized_std)
    num_channels = tensor.shape[1]
    kernel = kernel[None, :].repeat(num_channels, 1, 1)
    kernel = kernel.to(tensor.device)
    padding_size = (kernel_size - 1) // 2
    convolved_tensor = F.conv1d(tensor, kernel, padding=padding_size, groups=num_channels)
    pmf_tensor = convolved_tensor / convolved_tensor.sum(dim=-1, keepdim=True)
    return pmf_tensor

def scene_gaussian_smooth(config, scene_prob, sigma):
    if config in ["center_single", "distribute_four", "distribute_nine"]:
        smoothed_position_prob = gaussian_smooth(scene_prob.position_prob, sigma)
        smoothed_number_prob = gaussian_smooth(scene_prob.number_prob, sigma)
        smoothed_type_prob = gaussian_smooth(scene_prob.type_prob, sigma)
        smoothed_size_prob = gaussian_smooth(scene_prob.size_prob, sigma)
        smoothed_color_prob = gaussian_smooth(scene_prob.color_prob, sigma)
        smoothed_scene_prob = SceneProb(smoothed_position_prob, smoothed_number_prob, 
                                        smoothed_type_prob, smoothed_size_prob, smoothed_color_prob)
    else:
        if config == "in_out_four":
            smoothed_scene_prob1 = scene_gaussian_smooth("distribute_four", scene_prob[0], sigma)
        else:
            smoothed_scene_prob1 = scene_gaussian_smooth("center_single", scene_prob[0], sigma)
        smoothed_scene_prob2 = scene_gaussian_smooth("center_single", scene_prob[1], sigma)
        smoothed_scene_prob = (smoothed_scene_prob1, smoothed_scene_prob2)
    return smoothed_scene_prob

def compute_loss_and_scores(outputs, candidates, targets, loss_fn, args):
    if "distribute" in args.config:
        position_loss = loss_fn(outputs.position, candidates.position[torch.arange(candidates.position.shape[0]), targets])
    else:
        position_loss = 0
    type_loss = loss_fn(outputs.type, candidates.type[torch.arange(candidates.type.shape[0]), targets])
    size_loss = loss_fn(outputs.size, candidates.size[torch.arange(candidates.size.shape[0]), targets])
    color_loss = loss_fn(outputs.color, candidates.color[torch.arange(candidates.color.shape[0]), targets])
    loss = (position_loss + type_loss + color_loss + size_loss)/4

    if "distribute" in args.config:
        position_score = loss_fn.score(outputs.position.reshape((outputs.position.shape[0], -1)).unsqueeze(1).repeat(1, 8, 1), 
                                        candidates.position)
    else:
        position_score = 0
    type_score = loss_fn.score(outputs.type.reshape((outputs.type.shape[0], -1)).unsqueeze(1).repeat(1, 8, 1), candidates.type)
    color_score = loss_fn.score(outputs.color.reshape((outputs.color.shape[0], -1)).unsqueeze(1).repeat(1, 8, 1), candidates.color)
    size_score = loss_fn.score(outputs.size.reshape((outputs.size.shape[0], -1)).unsqueeze(1).repeat(1, 8, 1), candidates.size)
    scores = (position_score + type_score + color_score + size_score)/4

    return loss, scores

def train(args, env, device):
    '''
    Training and validation of learnable NVSA backend
    '''
    def train_epoch(epoch):
        model.train()
        if args.config == "in_out_four":
            model2.train()

        # Define tracking meters
        loss_avg = AverageMeter('Loss', ':.3f')
        acc_avg = AverageMeter('Accuracy', ':.3f')

        for counter, (extracted, targets, all_action_rule) in enumerate(tqdm(train_loader)):
            extracted, targets, all_action_rule = extracted.to(device), targets.to(device), all_action_rule.to(device)

            exist_logprob, type_logprob, size_logprob, color_logprob = create_one_hot(extracted, args.config)
            model_output = [exist_logprob.to(device), type_logprob.to(device), size_logprob.to(device), color_logprob.to(device)]
            scene_prob, _ = env.prepare(model_output)  
            
            if args.sigma_smooth > 0:
                scene_prob = scene_gaussian_smooth(args.config, scene_prob, args.sigma_smooth)

            if args.config in ["center_single", "distribute_four", "distribute_nine"]:
                outputs, candidates = model(scene_prob)
                loss, scores = compute_loss_and_scores(outputs, candidates, targets, loss_fn, args)
            else:
                if args.config == "in_out_four":
                    outputs1, candidates1 = model2(scene_prob[0])
                else:
                    outputs1, candidates1 = model(scene_prob[0])
                outputs2, candidates2 = model(scene_prob[1])
                loss1, scores1 = compute_loss_and_scores(outputs1, candidates1, targets, loss_fn, args)
                loss2, scores2 = compute_loss_and_scores(outputs2, candidates2, targets, loss_fn, args)
                loss = (loss1 + loss2)/2
                scores = (scores1 + scores2)/2
            
            predictions = torch.argmax(scores, dim=-1)
            accuracy = ((predictions == targets).sum()/len(targets))*100

            loss_avg.update(loss.item(), extracted.size(0))
            acc_avg.update(accuracy.item(), extracted.size(0))
            
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients with l2 normalization
            if args.clip: 
                val = torch.nn.utils.clip_grad_norm_(parameters=train_param, max_norm=args.clip, norm_type=2.0)

            optimizer.step()

        print("Epoch {}, Total Iter: {}, Train Avg Loss: {:.6f}, Train Avg Accuracy: {:.6f}".format(epoch, 
                                                                                                    counter, loss_avg.avg, acc_avg.avg))

        writer.add_scalar('loss/training',loss_avg.avg,epoch)
        writer.add_scalar('accuracy/training',acc_avg.avg,epoch)
        return 

    def validate_epoch(epoch):
        model.eval()
        if args.config == "in_out_four":
            model2.eval()
        
        loss_avg = AverageMeter('Loss', ':.3f')
        acc_avg = AverageMeter('Accuracy', ':.3f')

        for counter, (extracted, targets, all_action_rule) in enumerate(tqdm(val_loader)):
            extracted, targets, all_action_rule = extracted.to(device), targets.to(device), all_action_rule.to(device)
            
            exist_logprob, type_logprob, size_logprob, color_logprob = create_one_hot(extracted, args.config)
            model_output = [exist_logprob.to(device), type_logprob.to(device), size_logprob.to(device), color_logprob.to(device)]
            scene_prob, _ = env.prepare(model_output)

            if args.sigma_smooth > 0:
                scene_prob = scene_gaussian_smooth(args.config, scene_prob, args.sigma_smooth)

            if args.config in ["center_single", "distribute_four", "distribute_nine"]:
                outputs, candidates = model(scene_prob)
                loss, scores = compute_loss_and_scores(outputs, candidates, targets, loss_fn, args)
            else:
                if args.config == "in_out_four":
                    outputs1, candidates1 = model2(scene_prob[0])
                else:
                    outputs1, candidates1 = model(scene_prob[0])
                outputs2, candidates2 = model(scene_prob[1])
                loss1, scores1 = compute_loss_and_scores(outputs1, candidates1, targets, loss_fn, args)
                loss2, scores2 = compute_loss_and_scores(outputs2, candidates2, targets, loss_fn, args)
                loss = (loss1 + loss2)/2
                scores = (scores1 + scores2)/2
            
            predictions = torch.argmax(scores, dim=-1)
            accuracy = ((predictions == targets).sum()/len(targets))*100
            
            loss_avg.update(loss.item(), extracted.size(0))
            acc_avg.update(accuracy.item(), extracted.size(0))

        print("Epoch {}, Valid Avg Loss: {:.6f}, Valid Avg Acc: {:.4f}".format(epoch, loss_avg.avg, acc_avg.avg))

        writer.add_scalar('loss/validation',loss_avg.avg,epoch)
        writer.add_scalar('accuracy/validation',acc_avg.avg,epoch)
        return acc_avg.avg
  
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(args.log_dir)

    # Init model
    model = AttributeLevelReasoner(args.device, args.config, model=args.model, hidden_layers=args.hidden_layers, 
                                   dictionary=args.backend_cb_discrete, vsa_conversion=args.vsa_conversion, 
                                   vsa_selection=args.vsa_selection, context_superposition=args.context_superposition)
    model.to(args.device)
    if args.config == "in_out_four":
        model2 = AttributeLevelReasoner(args.device, "distribute_four", model=args.model, hidden_layers=args.hidden_layers, 
                                        dictionary=args.backend_cb_discrete, vsa_conversion=args.vsa_conversion, 
                                        vsa_selection=args.vsa_selection, context_superposition=args.context_superposition)
        model2.to(args.device)
    

    # Init loss
    loss_fn = getattr(losses, args.loss_fn)()

    # Init optimizers
    train_param = list(model.parameters())
    if args.config == "in_out_four":
        train_param += list(model2.parameters())                      
    optimizer = optim.Adam(train_param, args.lr, weight_decay=args.weight_decay)

    # Load all checkpoints
    rule_path = os.path.join(args.resume,"checkpoint.pth.tar")
    if os.path.isfile(rule_path):
        checkpoint = torch.load(rule_path)
        model.load_state_dict(checkpoint['state_dict_model'])
        if args.config == "in_out_four":
            model2.load_state_dict(checkpoint['state_dict_model'])
        best_accuracy = checkpoint['best_accuracy']
        start_epoch=checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' at Epoch {:.3f}".format(rule_path,checkpoint["epoch"]))
    else:
        best_accuracy = 0
        start_epoch = 0
   
    # dataset loader
    train_set = RAVENDataset("train", args.data_dir, constellation_filter=args.config, rule_filter = args.gen_rule, attribute_filter = args.gen_attribute)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = RAVENDataset("val", args.data_dir, constellation_filter=args.config, rule_filter = args.gen_rule, attribute_filter = args.gen_attribute)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    # training loop starts
    for epoch in range(start_epoch,args.epochs):
        train_epoch(epoch)
        with torch.no_grad():
            accuracy = validate_epoch(epoch)
        
        # store model(s)
        is_best = accuracy > best_accuracy
        best_accuracy = max(accuracy,best_accuracy)
        
        if args.config == "in_out_four":
            save_checkpoint({'epoch': epoch + 1, 'state_dict_model': model.state_dict(), 'state_dict_model2': model2.state_dict(),
                    'best_accuracy': best_accuracy, 'optimizer' : optimizer.state_dict()}, is_best, savedir=args.checkpoint_dir)
        else:
            save_checkpoint({'epoch': epoch + 1, 'state_dict_model': model.state_dict(),
                    'best_accuracy': best_accuracy, 'optimizer' : optimizer.state_dict()}, is_best, savedir=args.checkpoint_dir)
    return writer

def test(args, env, device, writer=None, dset="RAVEN"):
    '''
    Testing of NVSA backend
    '''
    def test_epoch():
        model.eval()
        if args.config == "in_out_four":
            model2.eval()
        
        loss_avg = AverageMeter('Loss', ':.3f')
        acc_avg = AverageMeter('Accuracy', ':.3f')

        for counter, (extracted, targets, all_action_rule) in enumerate(tqdm(test_loader)):
            extracted, targets, all_action_rule = extracted.to(device), targets.to(device), all_action_rule.to(device)
            
            exist_logprob, type_logprob, size_logprob, color_logprob = create_one_hot(extracted, args.config)
            model_output = [exist_logprob.to(device), type_logprob.to(device), size_logprob.to(device), color_logprob.to(device)]
            scene_prob, _ = env.prepare(model_output)

            if args.sigma_smooth > 0:
                scene_prob = scene_gaussian_smooth(args.config, scene_prob, args.sigma_smooth)
            
            if args.config in ["center_single", "distribute_four", "distribute_nine"]:
                outputs, candidates = model(scene_prob)
                loss, scores = compute_loss_and_scores(outputs, candidates, targets, loss_fn, args)
            else:
                if args.config == "in_out_four":
                    outputs1, candidates1 = model2(scene_prob[0])
                else:
                    outputs1, candidates1 = model(scene_prob[0])
                outputs2, candidates2 = model(scene_prob[1])
                loss1, scores1 = compute_loss_and_scores(outputs1, candidates1, targets, loss_fn, args)
                loss2, scores2 = compute_loss_and_scores(outputs2, candidates2, targets, loss_fn, args)
                loss = (loss1 + loss2)/2
                scores = (scores1 + scores2)/2
            
            predictions = torch.argmax(scores, dim=-1)
            accuracy = ((predictions == targets).sum()/len(targets))*100
            
            loss_avg.update(loss.item(), extracted.size(0))
            acc_avg.update(accuracy.item(), extracted.size(0))

        # Save final result as npz (and potentially in Tensorboard)
        if not (writer is None):  
            writer.add_scalar("accuracy/testing-{}".format(dset), acc_avg.avg, 0)
            np.savez(args.save_dir+"result_{:}.npz".format(dset),loss = acc_avg.avg)
        else:
            args.save_dir = args.resume.replace("ckpt/","save/")
            np.savez(args.save_dir+"result_{:}.npz".format(dset),loss = acc_avg.avg)

        print("Test Avg Accuracy: {:.4f}".format(acc_avg.avg))

    # Load all checkpoint 
    model_path = os.path.join(args.resume,"model_best.pth.tar")
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        print("=> loaded checkpoint '{}' with accuracy {:.3f}".format(model_path,checkpoint["best_accuracy"]))
    else:
        raise ValueError("No checkpoint found at {:}".format(model_path)) 

    # Init the model
    model = AttributeLevelReasoner(args.device, args.config, model=args.model, hidden_layers=args.hidden_layers, 
                                   dictionary=args.backend_cb_discrete, vsa_conversion=args.vsa_conversion, 
                                   vsa_selection=args.vsa_selection, context_superposition=args.context_superposition)
    model.to(device)
    model.load_state_dict(checkpoint["state_dict_model"])
    if args.config == "in_out_four":
        model2 = AttributeLevelReasoner(args.device, "distribute_four", model=args.model, hidden_layers=args.hidden_layers, 
                                        dictionary=args.backend_cb_discrete, vsa_conversion=args.vsa_conversion, 
                                        vsa_selection=args.vsa_selection, context_superposition=args.context_superposition)
        model2.to(device)
        model2.load_state_dict(checkpoint["state_dict_model2"])

    # Init loss
    loss_fn = getattr(losses, args.loss_fn)()

    # Dataset loader
    test_set = RAVENDataset("test", args.data_dir,  constellation_filter=args.config, rule_filter = args.gen_rule, 
                            attribute_filter = args.gen_attribute)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    print("Evaluating on {}".format(args.config))
    with torch.no_grad():
        test_epoch()
        
    return writer

def main():
    arg_parser = argparse.ArgumentParser(description='NVSA lernable backend training and evaluation on RAVEN')

    arg_parser.add_argument("--mode", type=str, default="train", help="Train/test")
    arg_parser.add_argument("--exp_dir", type=str, default="results/")
    arg_parser.add_argument("--data_dir", type=str, default="dataset/")
    arg_parser.add_argument("--resume", type=str, default="", help="Resume from a initialized model")
    arg_parser.add_argument("--seed", type=int, default=1234, help="Random number seed")
    arg_parser.add_argument("--run", type=int, default=1, help="Run id")

    # Dataset 
    arg_parser.add_argument("--config", type=str, default="center_single", help="The configuration used for training")

    # Training hyperparameters
    arg_parser.add_argument("--model", type=str, default="MLP", help="Model used in the reasoner (MLP)")
    arg_parser.add_argument("--hidden_layers", type=int, default=3, help="Number of hidden layers to use in the neural model")
    arg_parser.add_argument("--epochs", type=int, default=50, help="The number of training epochs")
    arg_parser.add_argument("--batch_size", type=int, default=32, help="Size of batch")
    arg_parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    arg_parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay of optimizer, same as l2 reg")
    arg_parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loader")
    arg_parser.add_argument("--clip", type=float, default=10, help="Max value/norm in gradient clipping (now l2 norm)")
    arg_parser.add_argument("--vsa_conversion", action="store_true", default=False, help="Use or not the VSA converter")
    arg_parser.add_argument("--vsa_selection", action="store_true", default=False, help="Use or not the VSA selector")
    arg_parser.add_argument("--context_superposition", action="store_true", default=False, help="Create a key-value bindings superposion of the context VSA vectors")
    arg_parser.add_argument("--loss_fn", type=str, default="CosineLoss", help="Loss to use in the training")
    arg_parser.add_argument("--sigma_smooth", type=float, default=0, help="Standard deviation used in the gaussian smoothing to reproduce perception errors")
    arg_parser.add_argument('--gen_attribute',type=str, default="", help="Generalization experiment [Type, Size, Color]")
    arg_parser.add_argument('--gen_rule',type=str, default="", help="Generalization experiment [Arithmetic, Constant, Progression, Distribute_Three]")

    # NVSA backend settings
    arg_parser.add_argument('--nvsa-backend-d', type=int, default=1024, help="VSA dimension in backend" )
    arg_parser.add_argument('--nvsa-backend-k', type=int, default=4, help="Number of blocks in VSA vectors")
    args = arg_parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.cuda = torch.cuda.is_available()

    # Use a rng for reproducible results   
    rng = np.random.default_rng(seed=args.seed)

    # Load or define new codebooks
    backend_cb_cont, backend_cb_discrete = generate_nvsa_codebooks(args, rng)

    args.backend_cb_discrete = backend_cb_discrete
    args.backend_cb_cont = backend_cb_cont

    # backend for training/testing
    input_configs = ['center_single', 'left_right', 'up_down', 'in_out_single', 'distribute_four', 'in_out_four', 'distribute_nine']
    output_configs = ["center_single", "left_center_single_right_center_single", "up_center_single_down_center_single", "in_center_single_out_center_single", "distribute_four", "in_distribute_four_out_center_single", "distribute_nine"]
    configs_map= dict(zip(input_configs, output_configs))

    env = reasoning_env.get_env(configs_map[args.config],device)

    if args.mode == "train":
        args.exp_dir = args.exp_dir+"/attribute_level/{:}_model_{:}_hidden_layers_{:}_vsa_conv_{:}_vsa_sel_{:}_context_superpos_{:}_loss_fn_{:}_sigma_smooth_{:}_epochs_{:}_bs_{:}_lr_{:}_wd_{:}_d_{:}_k_{:}_attribute_{:}_rule_{:}/{:}/".format(
                                                    args.config, args.model, args.hidden_layers, args.vsa_conversion, args.vsa_selection, args.context_superposition,
                                                    args.loss_fn, args.sigma_smooth, args.epochs, args.batch_size, args.lr, args.weight_decay, args.nvsa_backend_d, args.nvsa_backend_k, args.gen_attribute, args.gen_rule, args.run)
        args.checkpoint_dir = args.exp_dir+"ckpt/"
        args.save_dir = args.exp_dir+"save/"
        args.log_dir = args.exp_dir+"log/"
        check_paths(args)

        # Run the actual training 
        writer = train(args, env, device)

        # Do final testing
        args.resume = args.checkpoint_dir
        writer = test(args, env, device, writer)
        
        writer.close()
    
    elif args.mode == "test":
        test(args, env, device)
 
if __name__ == "__main__":
    main()