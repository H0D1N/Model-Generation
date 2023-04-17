import os
import time
import shutil
import colossalai
import torch
from timm import utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.logging import get_dist_logger
from model.PrompBasedModel import PromptBasedModel
from loss.GateLoss import sparse_gate_loss
from data.cifarDataset import cifarDataset
from utils import ExponentialMovingAverage

parser=colossalai.get_default_parser()
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_ema',action='store_true')
parser.add_argument(
    "--model-ema-steps",
    type=int,
    default=32,
    help="the number of iterations that controls how often to update the EMA model (default: 32)",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.99,
    help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
)
parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")



def main():
    best_acc1 = 0


    colossalai.launch_from_torch(
    config= 'config.py',
    )

    writer = SummaryWriter(log_dir='AdaResnet_for_classfication')

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    logger.info("initialized distributed environment", ranks=[0])
    logger.info("=> creating model",ranks=[0])

    #Cifar100任务的Task不需要Embed
    model=PromptBasedModel(task_embed=False)

    model_ema=None
    if args.model_ema:
        adjust=args.world_size*gpc.config.BATCH_SIZES*args.model_ema_steps/gpc.config.NUM_EPOCHS
        alpha=1.0-args.model_ema_decay
        alpha=min(1,alpha*adjust)
        model_ema=ExponentialMovingAverage(model,device='cuda',decay=1-alpha)



    criterion = sparse_gate_loss()

    train_dataset = cifarDataset(train=True)
    test_dataset = cifarDataset( train=False)

    train_loader=get_dataloader(dataset=train_dataset,batch_size=gpc.config.BATCH_SIZES,shuffle=False,pin_memory=True,)
    test_loader=get_dataloader(dataset=test_dataset,batch_size=gpc.config.BATCH_SIZES,pin_memory=True,shuffle=False)

    optimizer=torch.optim.SGD(model.parameters(),lr=gpc.config.LEARNING_RATE,weight_decay=gpc.config.WEIGHT_DECAY
                             ,momentum=gpc.config.MOMENTUM)

    lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']),ranks=0)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,optimizer,criterion,train_loader,test_loader)

    logger.info("Engine is built", ranks=[0])

    if gpc.is_initialized(ParallelMode.PARALLEL_1D):
        scatter_gather = True
    else:
        scatter_gather = False

    if args.evaluate:
        validate(engine, test_dataloader,logger)
        return

    for epoch in range(args.start_epoch,gpc.config.NUM_EPOCHS):
        train(engine,train_loader,epoch,logger,model_ema)
        #EMA与无EMA作对比
        if model_ema:
            acc1,usage,loss=validate(model_ema,test_loader,logger,epoch,best_acc1,criterion=criterion)
            best_acc1 = max(acc1, best_acc1)
        acc1, usage, loss = validate(engine, test_loader, logger, epoch, best_acc1)
        lr_scheduler.step()

        writer.add_scalar(tag='acc',scalar_value=acc1,global_step=epoch)
        writer.add_scalar(tag='loss',scalar_value=loss,global_step=epoch)
        writer.add_scalar(tag='GateUsage',scalar_value=usage,global_step=epoch)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        #EMA的存储还没加入
        '''
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'Ada_ResNet',
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict()
        }, is_best)
        '''

def train(engine,train_loader,epoch,logger,model_ema=None):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    # switch to train mode
    engine.train()

    end = time.time()
    for i, (prompt, img, label) in enumerate(train_loader):
        # measure data loading time

        img=img.cuda()
        label=label.cuda()
        prompt=prompt.cuda()

        data_time.update(time.time() - end)
        engine.zero_grad()
        output,selection = engine(prompt,img)
        #output.data.masked_fill_((torch.ones_like(prompt).cuda()-prompt).bool(),0)
        output = output * prompt
        loss = engine.criterion(output,label,selection)
        losses.update(loss.item(), img.size(0))

        engine.backward(loss)
        engine.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(engine.model)


    logger.info(
        'Train: {}  '
        'Loss: {loss.avg:#.3g}  '
        'Time: {batch_time.avg:.3f}s  '
        'Data: {data_time.avg:.3f}  '.format(
                epoch,
                i,
                loss=losses,
                batch_time=batch_time,
                data_time=data_time),ranks=[0]
            )

def validate(engine,test_loader,logger,epoch,best_acc1,criterion=None):
    def run_validate(test_loader):
        with torch.no_grad():
            end = time.time()
            for i, (prompt,img, label) in enumerate(test_loader):

                img = img.cuda()
                label = label.cuda()
                prompt = prompt.cuda()

                output,selection = engine(prompt,img)
                #output.data.masked_fill_((torch.ones_like(prompt).cuda() - prompt).bool(), 0)
                #output.data.masked_fill_((torch.ones_like(prompt).cuda() - prompt).bool(), -1e4)
                output=output*prompt
                if criterion:
                    loss=criterion(output, label,selection)
                else:
                    loss = engine.criterion(output, label,selection)


                if selection is not None:
                    selection = selection.mean()
                    usage.update(selection.item(), img.size(0))

                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, label, topk=(1, 5))
                losses.update(loss.item(), img.size(0))
                top1.update(acc1.item(), img.size(0))
                top5.update(acc5.item(), img.size(0))


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


    batch_time = utils.AverageMeter()
    losses =utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    usage = utils.AverageMeter()

    # switch to evaluate mode
    engine.eval()
    run_validate(test_loader)
    logger.info(
        '{0}: {1}  '
        'Time:  {batch_time.avg:.3f}    '
        'Loss: {loss.avg:>6.4f}    '
        'Acc@1: {top1.avg:>7.4f}    '
        'Acc@5: {top5.avg:>7.4f}    '
        'Usage: {usage.avg:>7.4f}   '
        'Best: {best}  '.format(
            'Test',epoch,
            batch_time=batch_time,
            loss=losses,
            top1=top1,
            top5=top5,
            usage=usage,
            best=best_acc1), ranks=[0]
    )
    return top1.avg,usage.avg,losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    args = parser.parse_args()
    main()









