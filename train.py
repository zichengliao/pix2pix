"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.set_up(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    if len(model.gpu_ids) == 0:
        print('In CPU mode..')
    for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs
        epoch_start = time.time()  # timer for entire epoch
        epoch_iters = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()  # update learning rates in the beginning of every epoch.

        import warnings
        warnings.filterwarnings('ignore')
        for i, data in enumerate(dataset):  # inner loop within one epoch
            total_iters += opt.batch_size
            epoch_iters += opt.batch_size
            # iter_start_time = time.time()

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.__train__()  # calculate loss functions, get gradients, update network weights

            if opt.display_id > 0 and total_iters % opt.display_freq == 0:   # display images on visdom
                # model.compute_visuals()  # doesn't do anything except for colorization
                visualizer.plot_visuals(model.get_current_visuals())

            if epoch_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_losses(opt.epoch_from + epoch, epoch_iters, losses)
                if opt.display_id > 0:
                    visualizer.plot_losses(opt.epoch_from + epoch, float(epoch_iters) / dataset_size, losses)

        print('End of epoch %d / %d \t Time Taken: %d sec \\' % (opt.epoch_from + epoch,  opt.epoch_from + opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start))
        model.update_learning_rate()    # update lr after epoch optimization

        # model.compute_visuals()
        ##visualizer.display_current_results(model.get_current_visuals(), opt.epoch_from + epoch, True)
        visualizer.save_to_html(model.get_current_visuals(), epoch)  # save images to checkpoints/~/web
        if epoch % opt.save_epoch_freq == 0 and epoch < opt.n_epochs + opt.n_epochs_decay:  # cache our model every K epochs
            model.save_networks(opt.epoch_from + epoch)
            print('saved models at the end of epoch %d' % (opt.epoch_from + epoch))

    model.save_networks('latest')
    print('saved the latest models at the end of epoch %d' % (opt.epoch_from + epoch))
    print('Training process finished! Hooray~')
