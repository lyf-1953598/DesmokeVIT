import torch
import itertools

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import dark_channel_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'IC_A', 'DC_B', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'IC_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        # 训练模式是训练生成器和鉴别器，两组
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            #测试模式只有两个生成器
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionIC = networks.ICLoss()  # define inter-channel loss.
            self.criterionDC = dark_channel_loss.DCLoss() # define dark channel loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) 
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_IC_A = self.criterionIC(self.fake_B, False)
        self.loss_DC_A = self.criterionDC(self.fake_B)
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) + self.loss_IC_A + self.loss_DC_A

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_IC_B = self.criterionIC(self.fake_A, True)
        self.loss_DC_B = self.criterionDC(self.fake_A)
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) + self.loss_IC_B + self.loss_DC_B

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # lyf-perceptual/edge
        lambda_perceptual = 5.0  # Perceptual loss weight
        lambda_edge = 1.0  # Edge loss weight
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterionVGG = VGGLoss(device)
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        #lyf
        # Perceptual loss
        self.loss_perceptual_A = self.criterionVGG(self.rec_B.to(device), self.real_B.to(device)) * lambda_perceptual
        self.loss_perceptual_B = self.criterionVGG(self.rec_A.to(device), self.real_A.to(device)) * lambda_perceptual

        # Edge loss
        # self.loss_edge_A = edge_loss(self.rec_A, self.real_A, device) * lambda_edge
        # self.loss_edge_B = edge_loss(self.rec_B, self.real_B, device) * lambda_edge

        # print("loss_perceptual_A:",self.loss_perceptual_A)
        # print("loss_perceptualB:",self.loss_perceptual_B)
        # print("loss_edgeA:",self.loss_edge_A)
        # print("loss_edgeB:",self.loss_edge_B)
        # combined loss and calculate gradients
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_perceptual_A + self.loss_perceptual_B + self.loss_edge_A + self.loss_edge_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_perceptual_A + self.loss_perceptual_B
        self.loss_G.backward()
       

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


# 定义Perceptual Loss
# class VGGLoss(nn.Module):
#     def __init__(self, device):
#         super(VGGLoss, self).__init__()
#         vgg = models.vgg19(pretrained=True).features
#         self.vgg = nn.Sequential(*list(vgg.children())[:36]).to(device)
#         self.criterion = nn.L1Loss()

#     def forward(self, x, y):
#         x_vgg = self.vgg(x)
#         y_vgg = self.vgg(y)
#         return self.criterion(x_vgg, y_vgg)
# update in 20241125(chj)
class VGGLoss(nn.Module):
    def __init__(self, device, layer_indices=None, loss_type="L1"):
        super(VGGLoss, self).__init__()
        # 加载预训练的 VGG 模型
        vgg = models.vgg19(pretrained=True).features

        # 替换 ReLU(inplace=True) 为 ReLU(inplace=False)
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)

        # 默认提取的层索引
        if layer_indices is None:
            layer_indices = [3, 8, 15, 22, 29]  # ReLU_1_1, ReLU_2_1, ReLU_3_1, ReLU_4_1, ReLU_5_1
            # layer_indices = [2, 7, 12]
        self.layer_indices = layer_indices

        # 保存所需层
        self.selected_layers = nn.ModuleList([vgg[i] for i in layer_indices]).to(device)

        # 冻结参数
        for param in self.selected_layers.parameters():
            param.requires_grad = False

        # 损失函数
        self.criterion = nn.L1Loss() if loss_type == "L1" else nn.MSELoss()

    def forward(self, x, y):
        x_features, y_features = [], []
        for layer in self.selected_layers:
            x = layer(x)
            y = layer(y)
            x_features.append(x)
            y_features.append(y)

        # 计算损失（逐层累加）
        loss = sum(self.criterion(x_f, y_f) for x_f, y_f in zip(x_features, y_features))
        return loss



# 定义Edge Loss
def edge_loss(pred, target, device):
    laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]]).float().unsqueeze(0).unsqueeze(0).to(device)
    pred_edge = []
    target_edge = []
    for i in range(pred.size(1)):  # 对每个通道进行卷积
        pred_edge.append(F.conv2d(pred[:, i:i+1, :, :], laplacian_kernel, padding=1))
        target_edge.append(F.conv2d(target[:, i:i+1, :, :], laplacian_kernel, padding=1))
    pred_edge = torch.cat(pred_edge, dim=1)
    target_edge = torch.cat(target_edge, dim=1)
    return F.l1_loss(pred_edge, target_edge)

class SpatialConsistencyLoss(nn.Module):

    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def gradient(self, x):
        # 计算图像的梯度
        gradient_x = x[:, :, :-1, :] - x[:, :, 1:, :]
        gradient_y = x[:, :, :, :-1] - x[:, :, :, 1:]
        return gradient_x, gradient_y

    def forward(self, fake, real):
        fake_gradient_x, fake_gradient_y = self.gradient(fake)
        real_gradient_x, real_gradient_y = self.gradient(real)
        loss_x = self.l1_loss(fake_gradient_x, real_gradient_x)
        loss_y = self.l1_loss(fake_gradient_y, real_gradient_y)
        return loss_x + loss_y
