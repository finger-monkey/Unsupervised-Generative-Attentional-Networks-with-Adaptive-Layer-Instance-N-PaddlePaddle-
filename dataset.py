
from PIL import Image

import os
import os.path
import numpy as np
import random
import cv2
import numbers
import matplotlib
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images



#class RandomCrop(object):

    #def __init__(self, output_size):
        #if isinstance(output_size, int):
            #self.output_size = (output_size, output_size)
        #else:
            #self.output_size = output_size

    #def _get_params(self, img):
        #h, w, _ = img.shape
        #th, tw = self.output_size
        #if w == tw and h == th:
            #return 0, 0, h, w

        #i = random.randint(0, h - th)
        #j = random.randint(0, w - tw)
        #return i, j, th, tw

    #def __call__(self, img):
        #i, j, h, w = self._get_params(img)
        #cropped_img = img[i:i + h, j:j + w]
        #return cropped_img

def resize(img, size, interpolation=Image.BILINEAR):

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
    
class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        # if self.padding is not None:
        #     img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # # pad the width if needed
        # if self.pad_if_needed and img.size[0] < self.size[1]:
        #     img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # # pad the height if needed
        # if self.pad_if_needed and img.size[1] < self.size[0]:
        #     img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        top=i
        left=j
        height=h
        width=w
        #crop(img, top, left, height, width)
        return img.crop((left, top, left + width, top + height))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

def normalize(im, mean, std):
    im = im / 255.0
    mean=np.array(mean)
    mean = mean[:, None, None]
    std=np.array(std)
    std = std[:, None, None]
    im -= mean
    im /= std
    return im
def custom_reader(root,transforms=None):
    '''
    自定义reader
    '''
    def reader():
        samples = make_dataset(root)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root))
        for idx in range(len(samples)):
            path, target = samples[idx]
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img=img.transpose(Image.FLIP_LEFT_RIGHT)
            img=img.resize((256+30, 256+30))
            img=RandomCrop(256)(img)
            img=np.array(img)
            img=img.transpose((2, 0, 1))
            img=normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))            
            #if transforms is not None:
                #for t in transforms:
                    #img = t(img)
            yield img, target
    return reader
class DatasetFolder(object):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.niter=0
        self.length=len(self.samples)
        self.transform = transform
        self.target_transform = target_transform

    
    def __next__(self):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.niter<self.length:
            path, target = self.samples[self.niter]
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB') 
            img=img.transpose(Image.FLIP_LEFT_RIGHT)
            img=img.resize((256+30, 256+30))
            img=RandomCrop(256)(img)
            img=np.array(img)
            img=img.transpose((2, 0, 1))
            
            #mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            img=normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #img=img.resize((256+30, 256+30))
            #img=img.crop(256,256,256,256)
            #sample = self.loader(path)
            #sample = cv2.flip(sample,1,dst=None) #水平镜像
            #sample=cv2.resize(sample,(256+30,256+30))
            #sample=RandomCrop(256)(sample)
            #cv2.normalize(sample,sample)
            #if self.transform is not None:
                #sample = self.transform(img)
            #if self.target_transform is not None:
                #target = self.target_transform(target)
            
            self.niter += 1  
            #im = np.array(sample[0]).reshape(1, 3, 256,256).astype('float32')
            #a=im.reshape(256,256,-1)

            #a=cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
            #cv2.imwrite("1.jpg", a)         
            return img.reshape(1, 3, 256,256).astype('float32'), target  
        else:
            self.niter=0      
    def __iter__(self):
        return self
    #def __getitem__(self, index):
        #"""
        #Args:
            #index (int): Index

        #Returns:
            #tuple: (sample, target) where target is class_index of the target class.
        #"""
        #path, target = self.samples[index]
        #sample = self.loader(path)
        #if self.transform is not None:
            #sample = self.transform(sample)
        #if self.target_transform is not None:
            #target = self.target_transform(target)

        #return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return  np.array(img.convert('RGB'))


def default_loader(path):
    return pil_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
