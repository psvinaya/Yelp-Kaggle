import apollocaffe
# from apollocaffe.layers import NumpyData, Convolution, SoftmaxWithLoss, EuclideanLoss, Filler
from apollocaffe.layers import (Power, LstmUnit, Convolution, NumpyData,
                                Transpose, Filler, SoftmaxWithLoss,
                                Softmax, Concat, Dropout, InnerProduct, Accuracy)

from apollocaffe.models import googlenet, alexnet
import numpy as np
import pandas as pd
from PIL import Image
import sys
import os
import time

label_prefix = "predicted_labels_"
log_interval = 2
def load_dataset(n_imgs_train, n_imgs_val):
    train_file = "../data/processed/4/train_photo_to_attr.csv"
    val_file = "../data/processed/4/val_photo_to_attr.csv"
    dirname = "../data/zipped/train_photos/"

    def load_data(filename, n_imgs):
        data = pd.read_csv(filename)
        photo_ids = data.photo_id.tolist()
        y = np.array(data.attr.tolist(), dtype=np.int32)[0:n_imgs]
        X = np.zeros((n_imgs, 3, 224, 224), dtype=np.float64)
        for j, id in enumerate(photo_ids):
            if j >= n_imgs:
                break
            im = Image.open(os.path.join(dirname,''.join([str(id),'.jpg'])))
            width, height = im.size
            im = im.resize((224, 224), Image.ANTIALIAS)
            im_rgb = np.array(im.getdata(),
                    np.uint8).reshape(3, im.size[1], im.size[0])
            X[j] = im_rgb            
        return X, y

    X_train, y_train = load_data(train_file, n_imgs_train)
    X_val, y_val = load_data(val_file, n_imgs_val)

    return X_train, y_train, X_val, y_val

def generate_decapitated_googlenet(net):
    """Generates the googlenet layers until the inception_5b/output.
    The output feature map is then used to feed into the lstm layers."""

    google_layers = googlenet.googlenet_layers()
    google_layers[0].p.bottom[0] = "data"
    for layer in google_layers:
        # if layer.p.type in ["Convolution", "InnerProduct"]:
        #     for p in layer.p.param:
        #         p.lr_mult *= 4
        net.f(layer)
        if layer.p.name == "inception_5b/output":
            break

def simple_conv_train():
    net = apollocaffe.ApolloNet()
    f = open('/deep/u/vinaya/cs231n/Yelp-Kaggle/loss.txt', 'w')
    for i in range(1000):
        X_train, y_train, X_val, y_val = load_dataset(1, 1)
        net.clear_forward()
        net.f(NumpyData('data', X_train))
        net.f(NumpyData('label', y_train))
        net.f(Convolution('conv', (64,64), 1, bottoms=['data']))
        net.f(SoftmaxWithLoss('loss', bottoms=['conv', 'label']))
        net.backward()
        net.update(lr=0.1)
        if i % 100 == 0:
            print net.loss
    f.close()

def train(datasize, batchsize, numIter):
    batch_size = 100
    num_batches = 200
    net = apollocaffe.ApolloNet()
    # f = open('/deep/u/vinaya/cs231n/Yelp-Kaggle/loss.txt', 'a+')
    filler = Filler("gaussian", 0.005)
    X_train, y_train, X_val, y_val = load_dataset(datasize, 1)
    batch_size = batchsize
    num_batches = datasize / batchsize
    start = time.time()
    f = open('/deep/u/vinaya/cs231n/Yelp-Kaggle/loss_%d.txt'% datasize, 'w')
    for i in range(numIter):
        # X_train, y_train, X_val, y_val = load_dataset(20000, 1)
        predicted_labels = []
        for j in range(num_batches):
            X = X_train[j*batch_size:(j+1)*batch_size,:,:,:]
            y = y_train[j*batch_size:(j+1)*batch_size]
            net.clear_forward()
            net.f(NumpyData('data', X))
            net.f(NumpyData('label', y))
            # print "before google net"
            generate_decapitated_googlenet(net)
            net.f(InnerProduct('ip', 2,
                           bottoms=['inception_5b/output'], output_4d=True,
                           weight_filler=filler))
            ip_data = net.blobs['ip'].data
            net.f(SoftmaxWithLoss('loss', bottoms=['ip', 'label']))
            net.backward()
            net.update(lr=0.01)
            print >> f, 'epoch -', i, 'batch -', j
            print 'epoch -', i, 'batch -', j, 'loss - ', net.loss
            print >> f, 'loss', net.loss
            predicted_label = np.ravel(np.argmax(ip_data, axis = 1))
            predicted_labels.extend(predicted_label)
            # print 'predicted_label', np.ravel(np.argmax(ip_data, axis = 1))
            # print 'actual_label', y
        count = 0
        for idx, i in enumerate(predicted_labels):
            if i == y_train[idx]:
                count += 1
        print >> f, 'accuracy', count/float(len(y_train))
    print >> f, 'time taken:', (time.time() - start)
    f.close()

def main():
    """Sets up all the configurations for apollocaffe, and ReInspect
    and runs the trainer."""
    parser = apollocaffe.base_parser()
    parser.add_argument('--datasize', required=True)
    parser.add_argument('--batchsize', required=True)
    parser.add_argument('--numIter', required=True)
    args = parser.parse_args()
    # config = json.load(open(args.config, 'r'))
    # if args.weights is not None:
    #     config["solver"]["weights"] = args.weights
    # config["solver"]["start_iter"] = args.start_iter
    # apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    datasize = int(args.datasize)
    batchsize = int(args.batchsize)
    numIter = int(args.numIter)

    # apollocaffe.set_cpp_loglevel(args.loglevel)

    train(datasize, batchsize, numIter)

if __name__ == "__main__":
    main()
