import argparse
import PIL

import torch
from torch import Tensor
from torch.autograd import Variable
from torch import nn
from torch.optim import LBFGS

from torchvision.models import vgg19
import torchvision.transforms.functional as Fv


use_cuda = torch.cuda.is_available()


class FeatureExtractor(nn.Module):
    """
    Extract feature activations on provided network
    """
    def __init__(self, net, idxs):
        super(FeatureExtractor, self).__init__()
        self.net = net
        self.idxs = idxs

    def forward(self, img, detach=False):
        out = img
        Fs = []
        for i, layer in enumerate(self.net.children()):
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            if isinstance(layer, nn.MaxPool2d):  # for better gradients
                layer = nn.AvgPool2d(layer.kernel_size)
            out = layer(out)
            if i in self.idxs:
                if detach:
                    out = out.detach()
                Fs.append(out)
            if i == self.idxs[-1]:  # stop evaluating model on last used layer
                break
        return Fs


class StyleTransfer(nn.Module):
    def __init__(self, img_size, style_extractor, style_weights, content_extractor, content_weights):
        super(StyleTransfer, self).__init__()
        self.style_extractor = style_extractor
        self.style_weights = style_weights
        self.content_extractor = content_extractor
        self.content_weights = content_weights
        self.img_size = img_size

        self.mean = nn.Parameter(Tensor([0.485, 0.456, 0.406]))
        self.std = nn.Parameter(Tensor([0.229, 0.224, 0.225]))

        self.style_img_F = None
        self.style_img_G = None
        self.content_img_F = None

    @staticmethod
    def compute_features(img, extractor, detach=False):
        # batch shape is assumed to be 1
        fs = extractor(img, detach)
        fs = [f.view(1, f.size(1), -1) for f in fs]  # flatten width, height dimensions
        return fs

    @staticmethod
    def compute_gram_matrices(features, flatten=True):
        fs_sizes = [f.size(2) for f in features]
        gs = [torch.bmm(f, f.transpose(1, 2))/f_size for f, f_size in zip(features, fs_sizes)]
        if flatten:
            gs = [g.view(-1) for g in gs]
        return gs

    def extract_style(self, style_img):
        style_img = Fv.normalize(style_img, self.mean, self.std)[None]
        self.style_img_F = self.compute_features(style_img, self.style_extractor, detach=True)
        self.style_img_G = self.compute_gram_matrices(self.style_img_F)

    def extract_content(self, content_img):
        content_img = Fv.normalize(content_img, self.mean, self.std)[None]
        self.content_img_F = self.compute_features(content_img, self.content_extractor, detach=True)

    def forward(self, random_img):
        assert all((self.style_img_F, self.style_img_G, self.content_img_F)), 'you must provide style and content first'

        random_img = random_img - self.mean[:, None, None]
        random_img = random_img / self.std[:, None, None]
        random_img = random_img[None]

        # style
        random_img_F = self.compute_features(random_img, self.style_extractor)
        random_img_G = self.compute_gram_matrices(random_img_F)
        style_losses = [torch.nn.MSELoss()(Gr, Gs)*u for Gs, Gr, u in zip(self.style_img_G, random_img_G,
                                                                          self.style_weights)]
        L_style = sum(style_losses)
        L_style = L_style.view(-1)

        # content
        random_img_F = self.compute_features(random_img, self.content_extractor)
        content_losses = [torch.nn.MSELoss()(Fr, Fc)*u for Fc, Fr, u in zip(self.content_img_F, random_img_F,
                                                                            self.content_weights)]
        L_content = sum(content_losses)
        L_content = L_content.view(-1)
        return L_content, L_style

    @staticmethod
    def transform_from_pil(img_raw, size):
        img = Fv.to_tensor(Fv.resize(img_raw, size))
        if use_cuda:
            img = img.cuda()
        return img

    @staticmethod
    def transform_to_pil(img):
        img = img.detach()
        img.data.clamp_(0., 1.)
        img = Fv.to_pil_image(img.cpu())
        return img

    def transfer(self, content_img_raw, style_img_raw, n_iter, alpha, beta, size, print_every=50):
        content_img = self.transform_from_pil(content_img_raw, size)
        style_img = self.transform_from_pil(style_img_raw, size)
        random_img = Variable(content_img.clone(), requires_grad=True)
        random_img.data.clamp_(0., 1.)

        self.extract_content(content_img)
        self.extract_style(style_img)

        optimizer = LBFGS([random_img])
        itr = [0]
        while itr[0] <= n_iter:
            def closure():
                optimizer.zero_grad()
                Lc, Ls = self(random_img)
                Lc, Ls = Lc*alpha, Ls*beta
                loss = Lc + Ls
                if not itr[0] % print_every:
                    print("i: %d, loss: %5.3f, content_loss: %5.3f, style_loss: %5.3f" % (
                          itr[0], loss.item(), Lc.item(), Ls.item()))
                loss.backward()
                itr[0] += 1
                return loss
            optimizer.step(closure)
            random_img.data.clamp_(0., 1.)

        return self.transform_to_pil(random_img)


def main(args):
    content = PIL.Image.open(args.content)
    style = PIL.Image.open(args.style)

    net = vgg19(pretrained=True).features.eval()
    for p in net.parameters():
        p.requires_grad = False
    if args.show:
        print(args)
        for i, c in enumerate(net.children()):
            print(i, c)
        return

    if args.style_indices:
        style_indices = args.style_indices
    else:
        style_indices = [1, 6, 11, 20, 29]  # indices of layers to use as style in sequential vgg.features

    if args.style_weights:
        style_weights = args.style_weights
    else:
        style_weights = [1 for i in style_indices]

    if args.content_indices:
        content_indices = args.content_indices
    else:
        content_indices = [22]  # indices of layers to use as content in sequential vgg.features
    if args.content_weights:
        content_weights = args.content_weights
    else:
        content_weights = [1 for i in content_indices]

    # Create features extractor
    style_extractor = FeatureExtractor(net, style_indices)
    content_extractor = FeatureExtractor(net, content_indices)

    style_transfer = StyleTransfer(args.size, style_extractor, style_weights, content_extractor, content_weights)
    if use_cuda:
        style_transfer = style_transfer.cuda()

    stylized_img = style_transfer.transfer(content, style, n_iter=args.n_iter, alpha=args.alpha, beta=args.beta,
                                           size=args.size)
    stylized_img.save(args.output)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--content', '-c', type=str, required=True)
    parser.add_argument('--style', '-s', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--size', type=int, nargs='+', default=512, help="Output image size")
    parser.add_argument('--alpha', type=float, default=1, help="Content loss strength")
    parser.add_argument('--beta', type=float, default=1e3, help="Style loss strength")
    parser.add_argument('--n_iter', type=int, default=300)

    parser.add_argument('--style_indices', nargs='+', type=int,
                        help="Indices of network layers to get style features")
    parser.add_argument('--style_weights', nargs='+', type=int)
    parser.add_argument('--content_indices', nargs='+', type=int,
                        help="Indices of network layers to get content features")
    parser.add_argument('--content_weights', nargs='+', type=int)

    parser.add_argument('--show', action='store_true',
                        help="Print network description for indices selection")

    args = parser.parse_args()

    main(args)
