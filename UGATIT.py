import time, itertools
import paddle
# from dataset import custom_reader
from dataset import ImageFolder
from utils import *
from networks import *
from glob import glob
import random
import os
import numpy

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        # if torch.backends.cudnn.enabled and self.benchmark_flag:
        #     print('set benchmark !')
        #     torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        # train_transform = None
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.Resize((self.img_size + 30, self.img_size+30)),
        #     RandomCrop(self.img_size),
        #     # transforms.RandomResizedCrop(self.img_size)
        #     # transforms.ToTensor(),
        #     transforms.Permute(to_rgb=True),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        #     # https://github.com/PaddlePaddle/Paddle/issues/26155
        #     # https://github.com/PaddlePaddle/hapi
        # ])
        # test_transform = None
        # test_transform = transforms.Compose([
        #     transforms.Resize((self.img_size, self.img_size)),
        #     # transforms.ToTensor(),
        #     transforms.Permute(to_rgb=True),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])

        # self.trainA = custom_reader(os.path.join( self.dataset, 'trainA'), train_transform)
        # self.trainB = custom_reader(os.path.join(self.dataset, 'trainB'), train_transform)
        # self.testA = custom_reader(os.path.join(self.dataset, 'testA'), test_transform)
        # self.testB = custom_reader(os.path.join( self.dataset, 'testB'), test_transform)
        
        self.trainA = ImageFolder(os.path.join(self.dataset, 'trainA'))
        self.trainB = ImageFolder(os.path.join(self.dataset, 'trainB'))
        self.testA = ImageFolder(os.path.join(self.dataset, 'testA'))
        self.testB = ImageFolder(os.path.join(self.dataset, 'testB'))
        # self.trainA_loader = fluid.io.shuffle(paddle.batch(self.trainA, batch_size=self.batch_size),buf_size=self.batch_size)
        # self.trainA_loader = fluid.io.shuffle(paddle.batch(self.trainA, batch_size=self.batch_size),buf_size=self.batch_size)
        # self.trainA_loader =paddle.shuffle(paddle.batch(self.trainA, batch_size=self.batch_size))
        # self.trainB_loader = fluid.io.shuffle(paddle.batch(self.trainB, batch_size=self.batch_size),buf_size=self.batch_size)#######
        # self.trainB_loader = paddle.shuffle(paddle.batch(self.trainB, batch_size=self.batch_size))
        # self.trainA_loader = paddle.batch(self.trainA,batch_size=1)
        # self.trainA_loader = paddle.batch(fluid.io.shuffle(self.trainA,3),batch_size=self.batch_size)
        # self.trainB_loader = paddle.batch(self.trainB,batch_size=1)
        # self.testA_loader = paddle.batch(self.testA, batch_size=1)
        # self.testB_loader = paddle.batch(self.testB, batch_size=1)
        self.trainA_loader = self.trainA
        self.trainB_loader = self.trainB
        self.testA_loader = self.testA
        self.testB_loader = self.testB
        print('self.trainA_loader=',self.trainA_loader)

        # self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        # self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        # self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        # self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)



    def train(self):
        # place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        place = fluid.CUDAPlace(0)
        with fluid.dygraph.guard(place):

            """ Define Generator, Discriminator """
            self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
            self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
            self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
            self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
            self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
            self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            """ Define Loss """
            self.L1_loss = paddle.fluid.dygraph.L1Loss(reduction='mean')
            self.MSE_loss =  paddle.fluid.dygraph.MSELoss(reduction='mean')
            self.BCE_loss = BCEWithLogitsLoss()
            """ Trainer """
            self.G_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=(self.genA2B.parameters() + self.genB2A.parameters()), regularization=paddle.fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))
            # self.G_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), regularization=paddle.fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))
            # self.D_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=itertools.chain(self.disGA.parameters(), self.disGB.parameters(),self.disLA.parameters(), self.disLB.parameters()), regularization=paddle.fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))
            self.D_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=(self.disGA.parameters() + self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters()), regularization=paddle.fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))

            """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
            self.Rho_clipper = RhoClipper(0, 1)

            start_iter = 1
            if self.resume:
                model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
                if not len(model_list) == 0:
                    model_list.sort()
                    start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                    print('-------self.resume---------self.load:')
                    self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                    print(" [*] Load SUCCESS")
                    if self.decay_flag and start_iter > (self.iteration // 2):
                        self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                        self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

            # training loop
            print('training start !')
            start_time = time.time()
            for step in range(start_iter, self.iteration + 1):
                # if self.decay_flag and step > (self.iteration // 2):
                    # self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                    # self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

                # try:
                #     real_A = next(trainA_iter)  #########################

                # except:
                #     trainA_iter = iter(self.trainA_loader())
                #     real_A = next(trainA_iter)


                # try:
                #     real_B = next(trainB_iter) 
                # except:
                #     trainB_iter = iter(self.trainB_loader())
                #     real_B = next(trainB_iter)

                try:
                    real_A, _ = next(trainA_iter)  #########################

                except:
                    trainA_iter = self.trainA_loader
                    real_A, _ = next(trainA_iter)


                try:
                    real_B, _ = next(trainB_iter)
                except:
                    trainB_iter = self.trainB_loader
                    real_B, _ = next(trainB_iter)

                # real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                real_A = fluid.dygraph.base.to_variable(real_A)
                real_B = fluid.dygraph.base.to_variable(real_B)

                # real_A = paddle.fluid.layers.transpose(real_A,perm=[0,3,1,2])#batch,h,w,c转成c,h,w
                # real_B = paddle.fluid.layers.transpose(real_B,perm=[0,3,1,2])

                # Update D
                # self.D_optim.zero_grad()
                # https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/optimizer_cn/AdamOptimizer_cn.html#adamoptimizer
                # self.D_optim.clear_gradients()#########################################


                fake_A2B, _, _ = self.genA2B(real_A)    ################
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
                #https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn/ones_like_cn.html#ones-like
                D_ad_loss_GA = self.MSE_loss(real_GA_logit, paddle.fluid.layers.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit,paddle.fluid.layers.ones_like(fake_GA_logit))

                D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit,paddle.fluid.layers.ones_like(real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit,paddle.fluid.layers.ones_like(fake_GA_cam_logit))

                D_ad_loss_LA = self.MSE_loss(real_LA_logit,paddle.fluid.layers.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit,paddle.fluid.layers.ones_like(fake_LA_logit))

                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit,paddle.fluid.layers.ones_like(real_LA_cam_logit)) +  self.MSE_loss(fake_LA_cam_logit,paddle.fluid.layers.ones_like(fake_LA_cam_logit))

                D_ad_loss_GB = self.MSE_loss(real_GB_logit,paddle.fluid.layers.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit,paddle.fluid.layers.ones_like(fake_GB_logit))

                D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit,paddle.fluid.layers.ones_like(real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit,paddle.fluid.layers.ones_like(fake_GB_cam_logit))

                D_ad_loss_LB = self.MSE_loss(real_LB_logit,paddle.fluid.layers.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit,paddle.fluid.layers.ones_like(fake_LB_logit))

                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit,paddle.fluid.layers.ones_like(real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit,paddle.fluid.layers.ones_like(fake_LB_cam_logit))


                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B
                print('Discriminator_loss=',Discriminator_loss)
                Discriminator_loss.backward()
                # self.D_optim.step()
                self.D_optim.minimize(Discriminator_loss)
                # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,但是不绝对，
                # 可以根据具体的需求来做。只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。

                # Update G
                # self.G_optim.zero_grad()
                self.G_optim.clear_gradients()

                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)


                G_ad_loss_GA = self.MSE_loss(fake_GA_logit,paddle.fluid.layers.ones_like(fake_GA_logit))
                G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit,paddle.fluid.layers.ones_like(fake_GA_cam_logit))
                G_ad_loss_LA = self.MSE_loss(fake_LA_logit,paddle.fluid.layers.ones_like(fake_LA_logit))
                G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit,paddle.fluid.layers.ones_like(fake_LA_cam_logit))
                G_ad_loss_GB = self.MSE_loss(fake_GB_logit,paddle.fluid.layers.ones_like(fake_GB_logit))
                G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit,paddle.fluid.layers.ones_like(fake_GB_cam_logit))
                G_ad_loss_LB = self.MSE_loss(fake_LB_logit,paddle.fluid.layers.ones_like(fake_LB_logit))
                G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit,paddle.fluid.layers.ones_like(fake_LB_cam_logit))

                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit,paddle.fluid.layers.ones_like(fake_B2A_cam_logit)) + self.BCE_loss(fake_A2A_cam_logit, paddle.fluid.layers.ones_like(fake_A2A_cam_logit))
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, paddle.fluid.layers.ones_like(fake_A2B_cam_logit)) + self.BCE_loss(fake_B2B_cam_logit, paddle.fluid.layers.ones_like(fake_B2B_cam_logit))


                G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                # self.G_optim.step()
                self.G_optim.minimize(Generator_loss)

                # clip parameter of AdaILN and ILN, applied after optimizer step
                # self.genA2B.apply(self.Rho_clipper)
                # self.genB2A.apply(self.Rho_clipper)

                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    B2A = np.zeros((self.img_size * 7, 0, 3))

                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    for _ in range(train_sample_num):


                        try:
                            real_A, _ = next(trainA_iter)  #########################

                        except:
                            trainA_iter = self.trainA_loader
                            real_A, _ = next(trainA_iter)


                        try:
                            real_B, _ = next(trainB_iter)
                        except:
                            trainB_iter = self.trainB_loader
                            real_B, _ = next(trainB_iter)
        
                        # real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                        real_A = fluid.dygraph.base.to_variable(real_A)##########
                        real_B = fluid.dygraph.base.to_variable(real_B)#############################

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
                        # print('real_A[0]',real_A[0].shape)
                        # print('fake_A2A_heatmap[0]',fake_A2A_heatmap[0].shape)
                        # print('fake_A2A[0]',fake_A2A[0].shape)
                        # print('fake_A2B_heatmap[0]',fake_A2B_heatmap[0].shape)
                        # print('fake_A2B[0]',fake_A2B[0].shape)
                        # print('fake_A2B2A_heatmap[0]',fake_A2B2A_heatmap[0].shape)
                        # print('fake_A2B2A[0]',fake_A2B2A[0].shape)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    for _ in range(test_sample_num):
                        try:
                            real_A, _= next(testA_iter)
                        except:
                            testA_iter = (self.testA_loader)
                            real_A, _= next(testA_iter)

                        try:
                            real_B, _= next(testB_iter)
                        except:
                            testB_iter = (self.testB_loader)
                            real_B, _= next(testB_iter)
                        # real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                        real_A = fluid.dygraph.base.to_variable(real_A)
                        real_B = fluid.dygraph.base.to_variable(real_B)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

                # if step % 1000 == 0:
                # if step % 2 == 0:
                    # self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)
                    
                    # params = {}
                    # params['genA2B'] = self.genA2B.state_dict()
                    # params['genB2A'] = self.genB2A.state_dict()
                    # params['disGA'] = self.disGA.state_dict()
                    # params['disGB'] = self.disGB.state_dict()
                    # params['disLA'] = self.disLA.state_dict()
                    # params['disLB'] = self.disLB.state_dict()
                    # torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
                    # fluid.save_dygraph(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
                    # fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(dir, self.dataset + '_genA2B_latest'))
                    # fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(dir, self.dataset + '_genB2A_latest'))
                    # fluid.save_dygraph(self.disGA.state_dict(), os.path.join(dir, self.dataset + '_disGA_latest'))
                    # fluid.save_dygraph(self.disGB.state_dict(), os.path.join(dir, self.dataset + '_disGB_latest'))
                    # fluid.save_dygraph(self.disLA.state_dict(), os.path.join(dir, self.dataset + '_disLA_latest'))
                    # fluid.save_dygraph(self.disLB.state_dict(), os.path.join(dir, self.dataset + '_disLB_latest'))


    def save(self, dir, step):
        # params = {}
        # params['genA2B'] = self.genA2B.state_dict()
        # params['genB2A'] = self.genB2A.state_dict()
        # params['disGA'] = self.disGA.state_dict()
        # params['disGB'] = self.disGB.state_dict()
        # params['disLA'] = self.disLA.state_dict()
        # params['disLB'] = self.disLB.state_dict()
        # torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        # fluid.save_dygraph(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

            # if step % self.save_freq == 0:
            #         self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

        # 存储模型
        if step % 500 == 0:
            fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(dir, self.dataset + '_genA2B_latest'))
            fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(dir, self.dataset + '_genB2A_latest'))
            fluid.save_dygraph(self.disGA.state_dict(), os.path.join(dir, self.dataset + '_disGA_latest'))
            fluid.save_dygraph(self.disGB.state_dict(), os.path.join(dir, self.dataset + '_disGB_latest'))
            fluid.save_dygraph(self.disLA.state_dict(), os.path.join(dir, self.dataset + '_disLA_latest'))
            fluid.save_dygraph(self.disLB.state_dict(), os.path.join(dir, self.dataset + '_disLB_latest'))
        else:
            fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(dir, self.dataset + '_genA2B%07d' % step))
            fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(dir, self.dataset + '_genB2A%07d' % step))
            fluid.save_dygraph(self.disGA.state_dict(), os.path.join(dir, self.dataset + '_disGA%07d' % step))
            fluid.save_dygraph(self.disGB.state_dict(), os.path.join(dir, self.dataset + '_disGB%07d' % step))
            fluid.save_dygraph(self.disLA.state_dict(), os.path.join(dir, self.dataset + '_disLA%07d' % step))
            fluid.save_dygraph(self.disLB.state_dict(), os.path.join(dir, self.dataset + '_disLB%07d' % step))
        print('-----存储6个模型----')

    def load(self, dir, step=None,isTest=False):
        print('-----加-----加载模型----')
        if isTest==True:
            p_genA2B=fluid.load_dygraph(os.path.join(dir, self.dataset + '_genA2B_latest'))
            p_genB2A=fluid.load_dygraph(os.path.join(dir, self.dataset + '_genB2A_latest'))
            p_disGA=fluid.load_dygraph(os.path.join(dir, self.dataset + '_disGA_latest'))
            p_disGB=fluid.load_dygraph(os.path.join(dir, self.dataset + '_disGB_latest'))
            p_disLA=fluid.load_dygraph(os.path.join(dir, self.dataset + '_disLA_latest'))
            p_disLB=fluid.load_dygraph(os.path.join(dir, self.dataset + '_disLB_latest'))

        else:
            p_genA2B=fluid.load_dygraph(os.path.join(dir, self.dataset + '_genA2B%07d' % step))
            p_genB2A=fluid.load_dygraph(os.path.join(dir, self.dataset + '_genB2A%07d' % step))
            p_disGA=fluid.load_dygraph(os.path.join(dir, self.dataset + '_disGA%07d' % step))
            p_disGB=fluid.load_dygraph(os.path.join(dir, self.dataset + '_disGB%07d' % step))
            p_disLA=fluid.load_dygraph(os.path.join(dir, self.dataset + '_disLA%07d' % step))
            p_disLB=fluid.load_dygraph(os.path.join(dir, self.dataset + '_disLB%07d' % step))


        # params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(p_genA2B)
        self.genB2A.load_state_dict(p_genB2A)
        self.disGA.load_state_dict(p_disGA)
        self.disGB.load_state_dict(p_disGB)
        self.disLA.load_state_dict(p_disLA)
        self.disLB.load_state_dict(p_disLB)
        print('----加载模型完毕----')



    def test(self):
        print('--------test()---------')
        place = fluid.CUDAPlace(0)
        with fluid.dygraph.guard(place):

            # model_list = glob(os.path.join(self.result_dir, self.dataset, 'model'))
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
            # print('model_list=',model_list)

            if not len(model_list) == 0:
                """ Define Generator, Discriminator """
                self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
                self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
                # print('self.genA2B',self.genA2B)
                # self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
                # self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
                # self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
                # self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
                # model_list.sort()
                # # print('model_list[-1].split(' ')[-1]=',model_list[-1].split('_')[-1])#latest.pdparams
                # # print('=',model_list[-1].split('_')[-1].split('.')[0])#latest
                # iter = int(model_list[-1].split('_')[-1].split('.')[0])
                print('-------test---------self.load:')
                # self.load(os.path.join(self.result_dir, self.dataset, 'model'), isTest=True)
                p_genA2B,_=fluid.load_dygraph('results/data/model/data_genA2B_latest.pdparams')
                # print('-------------p_genA2B=',p_genA2B)
                p_genB2A,_=fluid.load_dygraph('results/data/model/data_genB2A_latest.pdparams')
                # p_disGA=fluid.load_dygraph('results/data/model/data_disGA_latest.pdparams')
                # p_disGB=fluid.load_dygraph('results/data/model/data_disGB_latest.pdparams')
                # p_disLA=fluid.load_dygraph('results/data/model/data_disLA_latest.pdparams')
                # p_disLB=fluid.load_dygraph('results/data/model/data_disLB_latest.pdparams')




                self.genA2B.load_dict(p_genA2B)
                self.genB2A.load_dict(p_genB2A)
                # self.disGA.set_dict(p_disGA)
                # self.disGB.set_dict(p_disGB)
                # self.disLA.set_dict(p_disLA)
                # self.disLB.set_dict(p_disLB)
                # paddle.fluid.dygraph.Layer.set_dict(stat_dict, include_sublayers=True)
                self.genA2B.eval(), self.genB2A.eval()



                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            for n, (real_A, _) in enumerate(self.testA_loader):
                # real_A = real_A.to(self.device)
                real_A = fluid.dygraph.base.to_variable(real_A)

                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

                A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                    cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                    cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                    cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

            for n, (real_B, _) in enumerate(self.testB_loader):
                # real_B = real_B.to(self.device)
                real_B = fluid.dygraph.base.to_variable(real_B)


                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                    cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                    cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)

class RandomCrop(object):

    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def _get_params(self, img):
        h, w, _ = img.shape
        th, tw = self.output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self._get_params(img)
        cropped_img = img[i:i + h, j:j + w]
        return cropped_img

class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out


