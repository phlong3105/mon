# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import utility
import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    # def train(self):
    #     torch.set_grad_enabled(True)
    #     self.loss.step()
    #     self.model.eval()
    #     timer_data, timer_model = utility.timer(), utility.timer()
    #     #print(self.loader_train)
    #     for epoch in range(self.args.epochs):
    #         self.ckp.write_log('\nEpoch {}'.format(epoch + 1))
    #         self.loss.start_log()
    #         timer_data.tic()
    #         #print(self.loader_train)
    #         for batch, d in enumerate(self.loader_train):
    #             i=0
    #             for idx_scale, scale in enumerate(self.scale):
    #                 d.dataset.set_scale(idx_scale)
    #                 if self.args.derain:
    #                     for norain, rain, filename in tqdm(d, ncols=80):
    #                         norain, rain = self.prepare(norain, rain)
    #                         timer_data.hold()
    #                         timer_model.tic()
    #                         self.optimizer.zero_grad()
    #                         sr = self.model(rain, idx_scale)
    #                         loss = self.loss(sr, norain)
    #                         loss.backward()
    #                         self.optimizer.step()
    #                         timer_model.hold()

    #                         if (batch + 1) % self.args.print_every == 0:
    #                             self.ckp.write_log(
    #                                 '[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
    #                                     (batch + 1) * self.args.batch_size,
    #                                     len(self.loader_train.dataset),
    #                                     self.loss.display_loss(batch),
    #                                     timer_model.release(),
    #                                     timer_data.release()))

    #                         timer_data.tic()
    #                 elif self.args.denoise:
    #                     for hr, _,filename in tqdm(d, ncols=80):
    #                         hr = self.prepare(hr)[0]
    #                         noisy_level = self.args.sigma
    #                         noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
    #                         nois_hr = (noise+hr).clamp(0,255)
    #                         timer_data.hold()
    #                         timer_model.tic()
    #                         self.optimizer.zero_grad()
    #                         sr = self.model(nois_hr, idx_scale)
    #                         loss = self.loss(sr, hr)
    #                         loss.backward()
    #                         self.optimizer.step()
    #                         timer_model.hold()

    #                         if (batch + 1) % self.args.print_every == 0:
    #                             self.ckp.write_log(
    #                                 '[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
    #                                     (batch + 1) * self.args.batch_size,
    #                                     len(self.loader_train.dataset),
    #                                     self.loss.display_loss(batch),
    #                                     timer_model.release(),
    #                                     timer_data.release()))

    #                         timer_data.tic()
    #                 else:
    #                     for lr, hr, filename in tqdm(d, ncols=80):
    #                         lr, hr = self.prepare(lr, hr)
    #                         timer_data.hold()
    #                         timer_model.tic()
    #                         self.optimizer.zero_grad()
    #                         sr = self.model(lr, idx_scale)
    #                         loss = self.loss(sr, hr)
    #                         loss.backward()
    #                         self.optimizer.step()
    #                         timer_model.hold()

    #                         if (batch + 1) % self.args.print_every == 0:
    #                             self.ckp.write_log(
    #                                 '[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
    #                                     (batch + 1) * self.args.batch_size,
    #                                     len(self.loader_train.dataset),
    #                                     self.loss.display_loss(batch),
    #                                     timer_model.release(),
    #                                     timer_data.release()))

    #                         timer_data.tic()
    #                 i = i+1
            
    #         self.loss.end_log(len(self.loader_train))
    #         self.error_last = self.loss.log[-1, -1]
    #         self.optimizer.schedule()

    #         if self.args.save:
    #             self.ckp.save(self, epoch, is_best=(self.error_last < self.ckp.error_min))
    #         self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_model.toc()), refresh=True)

    def train(self):
        """Trainer"""


    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                if self.args.derain:
                    for norain, rain, filename in tqdm(d, ncols=80):
                        norain,rain = self.prepare(norain, rain)
                        sr = self.model(rain, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        
                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, norain, scale, self.args.rgb_range
                        ) 
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 1)
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                    isderain = 0
                elif self.args.denoise:
                    for hr, _,filename in tqdm(d, ncols=80):
                        hr = self.prepare(hr)[0]
                        noisy_level = self.args.sigma
                        noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
                        nois_hr = (noise+hr).clamp(0,255)
                        sr = self.model(nois_hr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr, nois_hr, hr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 50)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                else:
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        #import pdb
                        #pdb.set_trace()
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                        i = i+1
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
        
    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch.
        :param epoch: current epoch
        :type epoch: int
        :param lr: learning rate of cyclegan
        :type lr: float
        :param niter: number of epochs with the initial learning rate
        :type niter: int
        :param niter_decay: number of epochs to linearly decay learning rate to zero
        :type niter_decay: int
        """
        self.epoch = epoch
        value = self.args.decay.split('-')
        value.sort(key=int)
        milestones = list(map(int, value))
        print("*********** epoch: {} **********".format(epoch))
        lr = self.args.lr * self.args.gamma ** bisect_right(milestones, epoch)
        self.adjust_lr('model', self.optimizer, lr)
        print("*********************************")