'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 16 --adapt_method False --num_domains 1 --dataset PH2  --k_fold 4 > 4MedFormer_PH2.out 2>&1 &
'''
import argparse
from sqlite3 import adapt
import yaml
import os, time
from datetime import datetime

import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics
from torch.utils.tensorboard import SummaryWriter

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.losses import dice_loss
from Utils.pieces import DotDict

torch.cuda.empty_cache()

def main(config):
    # set gpu
    device_ids = range(torch.cuda.device_count())
    
    # prepare train, val, test datas
    train_loaders = {}  # initialize data loaders
    val_loaders = {}
    test_loaders = {}
    for dataset_name in config.data.name:
        datas = Dataset_wrap_csv(k_fold=config.data.k_fold, use_old_split=True, img_size=config.data.img_size, 
            dataset_name = dataset_name, split_ratio=config.data.split_ratio, 
            train_aug=config.data.train_aug, data_folder=config.data.data_folder)
        train_data, val_data, test_data = datas['train'], datas['test'], datas['test']

        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.train.batch_size,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
        train_loaders[dataset_name] = train_loader
        val_loaders[dataset_name] = val_loader
        test_loaders[dataset_name] = test_loader
        print('{} has {} training samples'.format(dataset_name, len(train_loader.dataset)))
    print('{} k_folder, {} val'.format(config.data.k_fold, config.data.use_val))

    
    # prepare model
    if config.model == 'SwinSeg':
        from Models.Transformer.Swin_adapters import SwinSimpleSeg_adapt
        model = SwinSimpleSeg_adapt(img_size=config.data.img_size,pretrained=True,pretrained_swin_name=config.swin.name,
                                pretrained_folder=config.pretrained_folder,
                              embed_dim=config.swin.EMBED_DIM,drop_path_rate=config.swin.DROP_PATH_RATE,
                              depths=config.swin.DEPTHS,num_heads=config.swin.NUM_HEADS,window_size=config.swin.WINDOW_SIZE,
                               debug=config.debug, adapt_method=False, num_domains=K)
        # freeze some parameters
        # for name, param in model.encoder.named_parameters():
        #     param.requires_grad = False
    elif config.model == 'SwinSeg_adapt':
        from Models.Transformer.Swin_adapters import SwinSimpleSeg_adapt
        model = SwinSimpleSeg_adapt(img_size=config.data.img_size,pretrained=True,pretrained_swin_name=config.swin.name,
                                pretrained_folder=config.pretrained_folder,
                              embed_dim=config.swin.EMBED_DIM,drop_path_rate=config.swin.DROP_PATH_RATE,
                              depths=config.swin.DEPTHS,num_heads=config.swin.NUM_HEADS,window_size=config.swin.WINDOW_SIZE,
                               debug=config.debug, adapt_method=config.model_adapt.adapt_method, num_domains=K)
        # freeze some parameters
        for name, param in model.encoder.named_parameters():
            if 'adapter' not in name and 'norm' not in name:
                param.requires_grad = False
    elif config.model == 'SwinSeg_CNNprompt_adapt':
        from Models.Transformer.Swin_adapters import SwinSimpleSeg_CNNprompt_adapt
        model = SwinSimpleSeg_CNNprompt_adapt(img_size=config.data.img_size,pretrained=True,pretrained_swin_name=config.swin.name,
                                pretrained_folder=config.pretrained_folder,
                              embed_dim=config.swin.EMBED_DIM,drop_path_rate=config.swin.DROP_PATH_RATE,
                              depths=config.swin.DEPTHS,num_heads=config.swin.NUM_HEADS,window_size=config.swin.WINDOW_SIZE,
                               debug=config.debug, adapt_method=config.model_adapt.adapt_method, num_domains=K)
        # freeze some parameters
        for name, param in model.encoder.named_parameters():
            if 'adapter' not in name and 'norm' not in name:
                param.requires_grad = False
    elif config.model == 'ViTSeg':
        from Models.Transformer.ViT_adapters import ViTSeg_adapt
        model = ViTSeg_adapt(pretrained=True, pretrained_vit_name=config.vit.name,
        pretrained_folder=config.pretrained_folder,img_size=config.data.img_size, patch_size=config.vit.patch_size,
        embed_dim=config.vit.embed_dim, depth=config.vit.depth, num_heads=config.vit.num_heads, 
        mlp_ratio=config.vit.mlp_ratio, drop_rate=config.vit.dropout_rate, 
        attn_drop_rate=config.vit.attention_dropout_rate, drop_path_rate=0.2, 
        debug=config.debug, adapt_method=False, num_domains=K)
        # for name, param in model.encoder.named_parameters():
        #     param.requires_grad = False
    elif config.model == 'ViTSeg_adapt':
        from Models.Transformer.ViT_adapters import ViTSeg_adapt
        model = ViTSeg_adapt(pretrained=True, pretrained_vit_name=config.vit.name,
        pretrained_folder=config.pretrained_folder,img_size=config.data.img_size, patch_size=config.vit.patch_size,
        embed_dim=config.vit.embed_dim, depth=config.vit.depth, num_heads=config.vit.num_heads, 
        mlp_ratio=config.vit.mlp_ratio, drop_rate=config.vit.dropout_rate, 
        attn_drop_rate=config.vit.attention_dropout_rate, drop_path_rate=0.2, 
        debug=config.debug, adapt_method=config.model_adapt.adapt_method, num_domains=K)
        for name, param in model.encoder.named_parameters():
            if 'adapter' not in name and 'norm' not in name:
                param.requires_grad = False
    elif config.model == 'ViTSeg_CNNprompt_adapt':
        from Models.Transformer.ViT_adapters import ViTSeg_CNNprompt_adapt
        model = ViTSeg_CNNprompt_adapt(pretrained=True, pretrained_vit_name=config.vit.name,
        pretrained_folder=config.pretrained_folder,img_size=config.data.img_size, patch_size=config.vit.patch_size,
        embed_dim=config.vit.embed_dim, depth=config.vit.depth, num_heads=config.vit.num_heads, 
        mlp_ratio=config.vit.mlp_ratio, drop_rate=config.vit.dropout_rate, 
        attn_drop_rate=config.vit.attention_dropout_rate, drop_path_rate=0.2, 
        debug=config.debug, adapt_method=config.model_adapt.adapt_method, num_domains=K)
        for name, param in model.encoder.named_parameters():
            if 'adapter' not in name and 'norm' not in name:
                param.requires_grad = False    
    elif config.model =='AdaptFormer':
        from Models.Transformer.AdapterFormer import AdaptFormer
        model = AdaptFormer(pretrained=True, pretrained_vit_name=config.vit.name,
        pretrained_folder=config.pretrained_folder,img_size=config.data.img_size, patch_size=config.vit.patch_size,
        embed_dim=config.vit.embed_dim, depth=config.vit.depth, num_heads=config.vit.num_heads, 
        mlp_ratio=config.vit.mlp_ratio, drop_rate=config.vit.dropout_rate, 
        attn_drop_rate=config.vit.attention_dropout_rate, drop_path_rate=0.2, 
        debug=config.debug, adapt_method=config.model_adapt.adapt_method, num_domains=K)
        for name, param in model.encoder.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False  
    elif config.model == 'DeiTSeg':
        from Models.Transformer.ViT_adapters import ViTSeg_adapt
        model = ViTSeg_adapt(pretrained=True, pretrained_vit_name=config.deit.name,
        pretrained_folder=config.pretrained_folder,img_size=config.data.img_size, patch_size=config.deit.patch_size,
        embed_dim=config.deit.embed_dim, depth=config.deit.depth, num_heads=config.deit.num_heads, 
        mlp_ratio=config.deit.mlp_ratio, drop_rate=config.deit.dropout_rate, 
        attn_drop_rate=config.deit.attention_dropout_rate, drop_path_rate=0.2, 
        debug=config.debug, adapt_method=False, num_domains=K)
        # for name, param in model.encoder.named_parameters():
        #     param.requires_grad = False
    elif config.model == 'DeiTSeg_CNNprompt_adapt':
        from Models.Transformer.ViT_adapters import ViTSeg_CNNprompt_adapt4
        model = ViTSeg_CNNprompt_adapt4(pretrained=True, pretrained_vit_name=config.deit.name,
        pretrained_folder=config.pretrained_folder,img_size=config.data.img_size, patch_size=config.deit.patch_size,
        embed_dim=config.deit.embed_dim, depth=config.deit.depth, num_heads=config.deit.num_heads, 
        mlp_ratio=config.deit.mlp_ratio, drop_rate=config.deit.dropout_rate, 
        attn_drop_rate=config.deit.attention_dropout_rate, drop_path_rate=0.2, 
        debug=config.debug, adapt_method=config.model_adapt.adapt_method, num_domains=K)
        for name, param in model.encoder.named_parameters():
            if 'adapter' not in name and 'norm' not in name:
                param.requires_grad = False      
    elif config.model == 'VPT':
        from Models.Transformer.ViT_prompt import ViTSeg_prompt
        model = ViTSeg_prompt(pretrained=True, pretrained_vit_name=config.vit.name,
        pretrained_folder=config.pretrained_folder,img_size=config.data.img_size, patch_size=config.vit.patch_size,
        embed_dim=config.vit.embed_dim, depth=config.vit.depth, num_heads=config.vit.num_heads, 
        mlp_ratio=config.vit.mlp_ratio, drop_rate=config.vit.dropout_rate, 
        attn_drop_rate=config.vit.attention_dropout_rate, drop_path_rate=0.2, 
        debug=config.debug, adapt_method=False, num_domains=K,prompt_len=100,prompt_drop_rate=0.1)  
        for name, param in model.encoder.named_parameters():
            if 'prompt' not in name:
                param.requires_grad = False 
    elif config.model == 'TransFuse':
        from Models.Hybrid_models.TransFuseFolder.TransFuse import TransFuse_L
        model = TransFuse_L(pretrained=True, pretrained_folder=config.pretrained_folder)
    elif config.model == 'FAT-Net':
        from Models.Hybrid_models.FAT_Net import FAT_Net
        model = FAT_Net()
    elif config.model == 'SwinUNETR':
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(img_size=(224,224), in_channels=3, out_channels=1, feature_size=48, use_checkpoint=False, spatial_dims=2)
    elif config.model == 'UNETR':
        from monai.networks.nets import UNETR 
        model = UNETR(img_size=224, in_channels=3, out_channels=1, spatial_dims=2)
    elif config.model == 'H2Former':
        from Models.Hybrid_models.H2FormerFolder.H2Former import res34_swin_MS
        model = res34_swin_MS(image_size=224,num_class=1,pretrained=True,pretrained_folder=config.pretrained_folder)
    elif config.model == 'UTNet':
        from Models.Hybrid_models.UTNetFolder.UTNet import UTNet
        model = UTNet(in_chan=3,base_chan=32,num_classes=1,reduce_size=config.data.img_size//32,block_list='1234',num_blocks=[1,1,1,1],
        num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    elif config.model == 'MedFormer':
        from Models.Hybrid_models.MedFormerFolder.MedFormer import MedFormer
        model = MedFormer(in_chan=3, num_classes=1, base_chan=32, conv_block='BasicBlock', conv_num=[2,0,0,0, 0,0,2,2], trans_num=[0,2,2,2, 2,2,0,0], 
        num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, fusion_dim=512, fusion_heads=16, map_size=3, 
        proj_type='depthwise', act=nn.GELU, expansion=2, attn_drop=0., proj_drop=0.)
    elif config.model == 'SwinUnet':
        from Models.Transformer.SwinUnet import SwinUnet
        model  = SwinUnet(img_size=config.data.img_size)




    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    
    # from thop import profile
    # input = torch.randn(1,3,224,224)
    # flops, params = profile(model, (input,))
    # print(f"total flops : {flops/1e9} G")

    # test model
    # x = torch.randn(5,3,224,224)
    # y = model(x)
    # print(y.shape)

    model = model.cuda()

    ## If multiple GPUs
    if len(device_ids) > 1: 
        model = torch.nn.DataParallel(model).cuda()
    
    criterion = [nn.BCELoss(), dice_loss]

    # only test
    if config.test.only_test == True:
        test(config, model, config.test.test_model_dir, test_loaders, criterion)
    else:
        train_val(config, model, train_loaders, val_loaders, criterion)
        test(config, model, best_model_dir, test_loaders, criterion)



# =======================================================================================================
def train_val(config, model, train_loaders, val_loaders, criterion):
    # optimizer loss
    if config.train.optimizer.mode == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    max_iou = 0 # use for record best model
    best_epoch = 0 # use for recording the best epoch
    # create training data loading iteration
    train_iters = {}
    for dataset_name in train_loaders.keys():
        train_iters[dataset_name] = iter(train_loaders[dataset_name])
    if config.train.num_iters:
        iterations = config.train.num_iters
    else:
        iterations = max([len(train_loaders[x]) for x in train_iters.keys()])
    
    torch.save(model.state_dict(), best_model_dir)
    for epoch in range(epochs):
        start = time.time()
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model.train()
        for train_step in range(epoch*iterations, (epoch+1)*iterations):
            # for each dataset, get one minibatch, get loss, sum all losses together
            # update once
            datas_loss_list = []  #record loss for datasets
            dice_train_list = []
            iou_train_list = []
            for dataset_name in config.data.name:
                try:
                    batch = next(train_iters[dataset_name])
                except StopIteration:
                    train_iters[dataset_name] = iter(train_loaders[dataset_name])
                    batch = next(train_iters[dataset_name])
                img = batch['image'].cuda().float()
                label = batch['label'].cuda().float()
                # domain_label = batch['set_id']
                if K == 1:
                    d = '0'
                else:
                    d = str(data2domain[dataset_name])
                # domain_label = torch.nn.functional.one_hot(domain_label, 4).float().cuda()
                other_models = set(['SwinUNETR','UNETR','SwinUnet','MedFormer','UTNet'])
                if config.model in other_models:
                    output = model(img)
                else:
                    output = model(img,d=d)['seg']

                output = torch.sigmoid(output)
    
                # calculate loss
                assert (output.shape == label.shape)
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss = sum(losses)
                datas_loss_list.append(loss)

                # calculate metrics
                with torch.no_grad():
                    output = output.cpu().numpy() > 0.5
                    label = label.cpu().numpy()
                    assert (output.shape == label.shape)
                    dice_train = metrics.dc(output, label)
                    iou_train = metrics.jc(output, label)
                    dice_train_list.append(dice_train)
                    iou_train_list.append(iou_train)

                # logging per batch
                # writer.add_scalar('Train/{}/BCEloss'.format(dataset_name), losses[0].item(), train_step)
                # writer.add_scalar('Train/{}/Diceloss'.format(dataset_name), losses[1].item(), train_step)
                writer.add_scalar('Train/{}/loss'.format(dataset_name), loss.item(), train_step)
                # writer.add_scalar('Train/{}/Di_score'.format(dataset_name), dice_train, train_step)
                writer.add_scalar('Train/{}/IOU'.format(dataset_name), iou_train, train_step)

            # backward
            multi_loss = sum(datas_loss_list)
            optimizer.zero_grad()
            multi_loss.backward()
            optimizer.step()

            # logging average per batch
            writer.add_scalar('Train/Average/sum_loss',multi_loss.item(), train_step)
            # writer.add_scalar('Train/Average/Di_score', sum(dice_train_list)/len(dice_train_list), train_step)
            writer.add_scalar('Train/Average/IOU', sum(iou_train_list)/len(iou_train_list), train_step)
            
            # end one training batch
            if config.debug: break

            # print
        print('Epoch {}, Total train step {} || sum_loss: {}, Avg Dice score: {}, Avg IOU: {}'.
        format(epoch, train_step, round(multi_loss.item(),5), round(sum(dice_train_list)/len(dice_train_list),4), 
        round(sum(iou_train_list)/len(iou_train_list),4)))
        print('Datasets: ', config.data.name, ' || loss: ', [round(x.item(), 4) for x in datas_loss_list], 
        ' || Dice score: ', [round(x, 4) for x in dice_train_list],
            ' || IOU: ', [round(x, 4) for x in iou_train_list])
            


        # -----------------------------------------------------------------
        # validate
        # ----------------------------------------------------------------
        model.eval()
        dice_val_list = []  # record results for each dataset
        iou_val_list = []
        loss_val_list = [] 
        # eval each dataset
        for dataset_name in config.data.name:
            dice_val_sum= 0
            iou_val_sum = 0
            loss_val_sum = 0
            num_val = 0
            for batch_id, batch in enumerate(val_loaders[dataset_name]):
                img = batch['image'].cuda().float()
                label = batch['label'].cuda().float()
                # domain_label = batch['set_id']
                if K == 1:
                    d = '0'
                else:
                    d = str(data2domain[dataset_name])
                # domain_label = torch.nn.functional.one_hot(domain_label, 4).float().cuda()
                batch_len = img.shape[0]

                with torch.no_grad():
                    other_models = set(['SwinUNETR','UNETR'])
                    if config.model in other_models:
                        output = model(img)
                    else:
                        output = model(img,d=d)['seg']
                    output = torch.sigmoid(output)

                    # calculate loss
                    assert (output.shape == label.shape)
                    losses = []
                    for function in criterion:
                        losses.append(function(output, label))
                    loss_val_sum += sum(losses)*batch_len

                    # calculate metrics
                    output = output.cpu().numpy() > 0.5
                    label = label.cpu().numpy()
                    dice_val_sum += metrics.dc(output, label)*batch_len
                    iou_val_sum += metrics.jc(output, label)*batch_len

                    num_val += batch_len
                    # end one val batch
                    if config.debug: break

            # logging per epoch for one dataset
            loss_val_epoch, dice_val_epoch, iou_val_epoch = loss_val_sum/num_val, dice_val_sum/num_val, iou_val_sum/num_val
            dice_val_list.append(dice_val_epoch)
            loss_val_list.append(loss_val_epoch.item())
            iou_val_list.append(iou_val_epoch)
            writer.add_scalar('Val/{}/loss'.format(dataset_name), loss_val_epoch.item(), epoch)
            writer.add_scalar('Val/{}/Di_score'.format(dataset_name), dice_val_epoch, epoch)
            writer.add_scalar('Val/{}/IOU'.format(dataset_name), iou_val_epoch, epoch)

        # logging average per epoch
        writer.add_scalar('Val/Average/sum_loss', sum(loss_val_list), epoch)
        writer.add_scalar('Val/Average/Di_score', sum(dice_val_list)/len(dice_val_list), epoch)
        writer.add_scalar('Val/Average/IOU', sum(iou_val_list)/len(iou_val_list), epoch)
        # print
        print('Epoch {}, Validation || sum_loss: {}, Avg Dice score: {}, Avg IOU: {}'.
                format(epoch, round(sum(loss_val_list),5), 
                round(sum(dice_val_list)/len(dice_val_list),4), round(sum(iou_val_list)/len(iou_val_list),4)))
        print('Datasets: ', config.data.name, ' || loss: ', [round(x, 4) for x in loss_val_list], 
        ' || Dice score: ', [round(x, 4) for x in dice_val_list],
         ' || IOU: ', [round(x, 4) for x in iou_val_list])


        # scheduler step, record lr
        writer.add_scalar('Lr', scheduler.get_last_lr()[0], epoch)
        scheduler.step()

        # store model using the average iou
        avg_val_iou_epoch = sum(iou_val_list)/len(iou_val_list)
        if avg_val_iou_epoch > max_iou:
            torch.save(model.state_dict(), best_model_dir)
            max_iou = avg_val_iou_epoch
            best_epoch = epoch
            print('New best epoch {}!==============================='.format(epoch))
        
        end = time.time()
        time_elapsed = end-start
        print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))

        # end one epoch
        if config.debug: return
    
    print('Complete training ---------------------------------------------------- \n The best epoch is {}'.format(best_epoch))

    return 




# ========================================================================================================
def test(config, model, model_dir, test_loaders, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_list = []  # record results for each dataset
    iou_test_list = []
    loss_test_list = [] 
    # test each dataset
    for dataset_name in config.data.name:
        dice_test_sum= 0
        iou_test_sum = 0
        loss_test_sum = 0
        num_test = 0
        for batch_id, batch in enumerate(test_loaders[dataset_name]):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            # domain_label = batch['set_id']
            if K == 1:
                d = '0'
            else:
                d = str(data2domain[dataset_name])
            batch_len = img.shape[0]
            other_models = set(['SwinUNETR','UNETR'])
            with torch.no_grad():
                if config.model in other_models:
                    output = model(img)
                else:
                    output = model(img,d=d)['seg']
                output = torch.sigmoid(output)

                # calculate loss
                assert (output.shape == label.shape)
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss_test_sum += sum(losses)*batch_len

                # calculate metrics
                output = output.cpu().numpy() > 0.5
                label = label.cpu().numpy()
                dice_test_sum += metrics.dc(output, label)*batch_len
                iou_test_sum += metrics.jc(output, label)*batch_len

                num_test += batch_len
                # end one test batch
                if config.debug: break

        # logging results for one dataset
        loss_test_epoch, dice_test_epoch, iou_test_epoch = loss_test_sum/num_test, dice_test_sum/num_test, iou_test_sum/num_test
        dice_test_list.append(dice_test_epoch)
        loss_test_list.append(loss_test_epoch.item())
        iou_test_list.append(iou_test_epoch)


    # logging average and store results
    dataset_name_list = config.data.name+['Total']
    loss_test_list.append(sum(loss_test_list))
    dice_test_list.append(sum(dice_test_list)/len(dice_test_list))
    iou_test_list.append(sum(iou_test_list)/len(iou_test_list))
    df = pd.DataFrame({
        'Name': dataset_name_list,
        'loss': loss_test_list,
        'Di_score': dice_test_list,
        'IOU': iou_test_list
    })
    df.to_csv(test_results_dir, index=False)

    # print
    print('========================================================================================')
    print('Test || Average loss: {}, Dice score: {}, IOU: {}'.
            format(round(sum(loss_test_list),5), 
            round(sum(dice_test_list)/len(dice_test_list),4), round(sum(iou_test_list)/len(iou_test_list),4)))
    print('Datasets: ', config.data.name, ' || loss: ', [round(x, 4) for x in loss_test_list], 
    ' || Dice score: ', [round(x, 4) for x in dice_test_list], ' || IOU: ', [round(x, 4) for x in iou_test_list])

    return




if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp_name', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str,default='Configs/multi_train_local.yml')
    parser.add_argument('--model', type=str,default='DeepResUnet')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='isic2018')
    parser.add_argument('--k_fold', type=str, default='No')
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['model'] = args.model
    config['train']['batch_size']=args.batch_size
    config['data']['name'] = args.dataset
    config['model_adapt']['adapt_method']=args.adapt_method
    config['model_adapt']['num_domains']=args.num_domains
    config['data']['k_fold'] = args.k_fold

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))
    
    store_config = config
    config = DotDict(config)
    
    # logging tensorbord, config, best model
    exp_dir = '{}/results/{}_{}_{}'.format(config.root_dir,args.exp_name,config.model,now.strftime("%Y%m%d_%H%M"))
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(exp_dir)
    best_model_dir = '{}/best.pth'.format(exp_dir)
    test_results_dir = '{}/test_results.csv'.format(exp_dir)

    # store yml file
    if config.debug == False:
        yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
    
    if config.model_adapt.num_domains != False:
        K = int(config.model_adapt.num_domains)
    else:
        from typing import List
        K = len(config.data.name) if isinstance(config.data.name, List) else 1 # num of domains
    print('K == {}'.format(K))
    data2domain = {config.data.name[i]:i for i in range(len(config.data.name))}

    main(config)
