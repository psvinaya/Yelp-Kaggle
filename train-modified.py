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

label_prefix = "/deep/u/vinaya/cs231n/yelp_shubham/predicted_labels/log_"
label_log_interval = 5
val_iter = 10
train_val_ratio = 5

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
        # net.update(lr=0.2, momentum=0.5, clip_gradients=0.1)

        if i % 100 == 0:
            print net.loss
    f.close()

def log_predicted_labels(iter, datasize, predicted_labels, y, is_val):
    if is_val:
        log_file = label_prefix + "val_" + str(datasize) + "_" + str(iter+1) + ".txt"
    else:
        log_file = label_prefix + "train_" + str(datasize) + "_" + str(iter+1) + ".txt" 

    if (iter+1)%label_log_interval == 0 or is_val:
        # log_file = label_prefix + str(datasize) + "_" + str(iter+1) + ".txt"
        result = np.column_stack((np.array(predicted_labels),y))
        np.savetxt(log_file, result, '%d', ",")

def forward_google_net(google_net, X, y):
    google_net.clear_forward()
    google_net.f(NumpyData('data', X))
    google_net.f(NumpyData('label', y))
    generate_decapitated_googlenet(google_net)

def forward_net(net, X, y):
    filler = Filler("gaussian", 0.005)

    net.clear_forward()
    net.f(NumpyData('data', X))
    net.f(NumpyData('label', y))
    net.f(InnerProduct('ip', 2, bottoms=['data'], output_4d=True,
       weight_filler=filler))
    net.f(SoftmaxWithLoss('loss', bottoms=['ip', 'label']))

def forward(google_net, net, X, y):
    forward_google_net(google_net, X, y)
    inp_net = google_net.blobs['inception_5b/output'].data
    forward_net(net, inp_net, y)
    ip_data = net.blobs['ip'].data
    predicted_label = np.ravel(np.argmax(ip_data, axis = 1))
    return predicted_label, net.loss

def getAccuracy(predicted, actual):
    count = 0
    for idx, i in enumerate(predicted):
        if i == actual[idx]:
            count += 1
    return count/float(len(actual))

def train(datasize, batchsize, numIter):
    # batch_size = 100
    # num_batches = 200
    google_net = apollocaffe.ApolloNet()
    net = apollocaffe.ApolloNet()
    val_datasize =  datasize/train_val_ratio
    X_train, y_train, X_val, y_val = load_dataset(datasize, val_datasize)
    batch_size = batchsize
    num_batches = datasize / batchsize
    num_val_batches = val_datasize / batchsize
    start = time.time()
    # f = open('/deep/u/vinaya/cs231n/yelp_shubham/loss_%d.txt'% datasize, 'w+')

    forward_google_net(google_net, np.expand_dims(X_train[0], axis = 0), [y_train[0]])
    google_net.load(googlenet.weights_file())

    for i in range(numIter):
        if (i+1)%val_iter == 0:
            f = open('/deep/u/vinaya/cs231n/yelp_shubham/loss_val_%d_%d.txt'% (datasize, numIter), 'a+')
            predicted_labels = []
            for j in range(num_val_batches):
                X = X_val[j*batch_size:(j+1)*batch_size,:,:,:]
                y = y_val[j*batch_size:(j+1)*batch_size]
                predicted_label, loss = forward(google_net, net, X, y)
                print >> f, 'val epoch -', (i+1)/val_iter, 'batch -', j
                print 'val epoch -', (i+1)/val_iter, 'batch -', j, 'loss - ', loss
                print >> f, 'val loss', loss
                predicted_labels.extend(predicted_label)

            log_predicted_labels(i, datasize, predicted_labels, y_val, True)
            accuracy = getAccuracy(predicted_labels, y_val)
            print >> f, 'val accuracy', accuracy
            f.close()

        f = open('/deep/u/vinaya/cs231n/yelp_shubham/loss_train_%d_%d.txt'% (datasize, numIter), 'a+')
        predicted_labels = []
        for j in range(num_batches):
            X = X_train[j*batch_size:(j+1)*batch_size,:,:,:]
            y = y_train[j*batch_size:(j+1)*batch_size]

            # forward_google_net(google_net, X, y)

            # inp_net = google_net.blobs['inception_5b/output'].data

            # forward_net(net, inp_net, y)
            # ip_data = net.blobs['ip'].data
            predicted_label, loss = forward(google_net, net, X, y)
            net.backward()
            net.update(lr=0.2)

            print >> f, 'epoch -', i, 'batch -', j
            print 'epoch -', i, 'batch -', j, 'loss - ', loss
            print >> f, 'training loss', loss
            # predicted_label = np.ravel(np.argmax(ip_data, axis = 1))
            predicted_labels.extend(predicted_label)
            # print 'predicted_label', np.ravel(np.argmax(ip_data, axis = 1))
            # print 'actual_label', y
        
        log_predicted_labels(i, datasize, predicted_labels, y_train, False)
        accuracy = getAccuracy(predicted_labels, y_train)
        print >> f, 'train accuracy', accuracy
        # count = 0
        # for idx, i in enumerate(predicted_labels):
        #     if i == y_train[idx]:
        #         count += 1
        # print >> f, 'accuracy', count/float(len(y_train))
        f.close()

    f = open('/deep/u/vinaya/cs231n/yelp_shubham/loss_%d.txt'% datasize, 'a+')
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
