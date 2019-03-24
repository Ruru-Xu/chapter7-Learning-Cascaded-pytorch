import os
import torch
import numpy as np
import sys
from datetime import datetime

from crowd_count import CrowdCounter
import network
from data_loader import ImageDataLoader
from timer import Timer
import utils
from evaluate_model import evaluate_model
from  logger import Logger

try:
    from termcolor import cprint
except ImportError:
    cprint = None
#
# try:
#     from pycrayon import CrayonClient
# except ImportError:
#     CrayonClient = None
CrayonClient = None

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)
        

logger=Logger('logs')
method = 'cmtl' #method name - used for saving model file
dataset_name = 'shtechA' #dataset name - used for saving model file
output_dir = 'dataset/Shanghai/saved_models/' #model files are saved here

#train and validation paths
train_path = 'dataset/Shanghai/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = 'dataset/Shanghai/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = 'dataset/Shanghai/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = 'dataset/Shanghai/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

#training configuration
start_step = 0
end_step = 2000
lr = 0.00001
momentum = 0.9
disp_interval = 500
log_interval = 250


#Tensorboard  config
use_tensorboard = True
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False   # remove all historical experiments in TensorBoardO
exp_name = None # the previous experiment name in TensorBoard



rand_seed = 64678    
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    
#loadt training and validation data
data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=False, pre_load=True)
class_wts = data_loader.get_classifier_weights()
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=False, pre_load=True)

#load net and initialize it
net = CrowdCounter(ce_weights=class_wts)
network.weights_normal_init(net, dev=0.01)
net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr) #优化算法

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

best_mae = sys.maxsize  #9223372036854775807

for epoch in range(start_step, end_step + 1):
    step = -1
    train_loss = 0
    for blob in data_loader:
        step = step + 1
        im_data = blob['data']   #image
        gt_data = blob['gt_density']   #
        gt_class_label = blob['gt_class_label'] #label

        # data augmentation on the fly
        if np.random.uniform() > 0.5:
            # randomly flip input image and density
            im_data = np.flip(im_data, 3).copy()  #图片
            gt_data = np.flip(gt_data, 3).copy()  #密度图
        if np.random.uniform() > 0.5:
            # add random noise to the input image
            im_data = im_data + np.random.uniform(-10, 10, size=im_data.shape)



        density_map = net(im_data, gt_data, gt_class_label, class_wts)  #output
        loss = net.loss     #loss
        train_loss += loss.item()
        step_cnt += 1
        optimizer.zero_grad()  #训练之前需要把所有参数的导数归零，也就是把loss关于weight的导数变成0.
        loss.backward()    # 计算得到loss后就要回传损失。要注意的是这是在训练的时候才会有的操作，测试时候只有forward过程
        # 回传损失过程中会计算梯度，然后需要根据这些梯度更新参数，optimizer.step()就是用来更新参数的。optimizer.step()后，
        # 你就可以从optimizer.param_groups[0][‘params’]里面看到各个层的梯度和权值信息。
        optimizer.step()  #有了更新好的导数，我们就可以更新我们的参数。这里是只更新一步

        if step % disp_interval == 0:
            duration = t.toc(average=False)  # 结束本次计时,返回值可以为多次的平均时间, 也可以为此次时间差toc(average=False)
            fps = step_cnt / duration  #每秒传输帧数(Frames Per Second)
            gt_count = np.sum(gt_data)
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            utils.save_results(im_data, gt_data, density_map, output_dir)
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                                                                                            step, 1. / fps, gt_count,
                                                                                            et_count)
            log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True

        if re_cnt:
            t.tic()  # 单次开始计时
            re_cnt = False

    if (epoch % 2 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
        network.save_net(save_name, net)
        # calculate error on the validation dataset
        mae, mse = evaluate_model(save_name, data_loader_val)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method, dataset_name, epoch)
        log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch, mae, mse)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae, best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
        if use_tensorboard:
            exp.add_scalar_value('MAE', mae, step=epoch)
            exp.add_scalar_value('MSE', mse, step=epoch)
            exp.add_scalar_value('train_loss', train_loss / data_loader.get_num_samples(), step=epoch)

        # 1. Log scalar values (scalar summary)
        info = { 'MAE': mae.item(), 'MSE': mse.item() } #这里只需要将loss和accuracy提供出来就行。注意这里不是tensor也不是numpy array而是单个的scalar

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)

        # 2. Log values and gradients of the parameters (histogram summary)#这里是针对所有的parameters和gradient来做histogram
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

        # im_data = torch.from_numpy(im_data)
        # # 3. Log training images (image summary)
        # info = { 'im_data': im_data.view(-1, 28, 28)[:10].cpu().numpy() }
        #
        # for tag, im_data in info.items():
        #     logger.image_summary(tag, im_data, step+1)
