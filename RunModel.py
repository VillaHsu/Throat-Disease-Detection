#!/usr/bin/env python
#IEEE_cGAN_classifierV02.py --savePath savePath --mode train --batch_size 128
import os, sys
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, merge
from keras.layers import Reshape, Input
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adagrad
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from PIL import Image
from sklearn import preprocessing
import argparse
import math
from data import load_LIDSet1, load_LIDSet1All
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pdb
from sklearn.utils import shuffle

class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        """ Constructor """
        pass

    def print_colored(self, string, color):
        """ Change color of string """
        return color + string + BColors.ENDC

col = BColors()

loss_alpha1=1.0
loss_alpha2=0.5
def categorical_crossentropy_withAlpha(y_true, y_pred):
    # Loss(Real/Fake) + alpha * Loss(Class)
    return loss_alpha1 * K.mean(K.binary_crossentropy(y_pred[:,0], y_true[:,0]), axis=-1) + loss_alpha2 * K.categorical_crossentropy(y_pred[:,1:51], y_true[:,1:51])

def accuracy_score(t, p):
    """
    Compute accuracy
    """
    return float(np.sum(p == t)) / len(p)
  
def train(BATCH_SIZE,savePath="output"):
    os.system("mkdir -p "+savePath+"/img")
    os.system("mkdir -p "+savePath+"/model")
    LDA_ON=0
    Flag_Generate=0
    if 1==1:
        data = load_LIDSet1All();
        #data = load_LIDSet1(10)
        n_classes=50
        feature_dim=49
        feature_dim_g=400
        LDA_ON=1
        X_train, Y_train = data['X_train'], data['y_train']
        X_dev, Y_dev = data['X_valid'], data['y_valid']
        X_test, Y_test = data['X_test'], data['y_test']        
        
    import IS2017_model_02 as model


    model.feature_dim=feature_dim
    model.n_classes=n_classes
    
    model.feature_dim_g=feature_dim_g
       
    if LDA_ON==1:
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, Y_train)
        X_train_lda=clf.transform(X_train)
        X_dev_lda=clf.transform(X_dev)
        X_test_lda=clf.transform(X_test)
  
    print("feature_dim:"+str(model.feature_dim))
    generator = model.generator_model()
    if 1==12:
        print("Using SGD optimizer")
        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    else:
        print("Using Adagrad optimizer")
        g_optim = Adagrad(lr=0.0005)
        d_optim = Adagrad(lr=0.0005)

    #categorical_crossentropy_withAlpha
    #categorical_crossentropy
    generator.compile(loss=categorical_crossentropy_withAlpha, optimizer=g_optim)
    generator.summary()
    
    discriminator = model.discriminator_model()
    discriminator.trainable = True
    discriminator.compile(loss=categorical_crossentropy_withAlpha, optimizer=d_optim) 
    discriminator.summary()
    
    discriminator_on_generator = model.generator_containing_discriminator(generator, discriminator)
    discriminator_on_generator.compile(loss=categorical_crossentropy_withAlpha, optimizer=g_optim)
    
    noise = np.zeros((BATCH_SIZE, 100))
    dev_best_cost=100
    test_best_cost=100
    for epoch in range(500):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        
        X_train_shuffle1, X_train_lda_shuffle, Y_train_shuffle1=shuffle(X_train,X_train_lda,Y_train)
        
        train_cost_avg=0
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
                
            image_batch_c   = X_train_shuffle1[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            image_batch_lda = X_train_lda_shuffle[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            image_batch_y   = Y_train_shuffle1[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            
            #pdb.set_trace()
            generated_images = generator.predict([image_batch_c,noise])
            # make one_hot vector for class labels
            y_onehot=np.zeros((BATCH_SIZE,n_classes))
            y_onehot[np.arange(BATCH_SIZE), image_batch_y] = 1
            y_onehot=y_onehot.reshape(-1,n_classes)
            y_onehot_all=np.concatenate((y_onehot,y_onehot),axis=0)
            
            #pdb.set_trace()            
            real_pairs = np.concatenate((image_batch_lda,image_batch_lda),axis=0) 
            fake_pairs = np.concatenate((image_batch_lda,generated_images),axis=0)
            images_all = np.concatenate((real_pairs,fake_pairs),axis=0)
            #labels_all=[1] * BATCH_SIZE + [0] * BATCH_SIZE

            #print(y_onehot_all.shape)
            real_fake_label= np.array([[1] * BATCH_SIZE + [0] * BATCH_SIZE])
            real_label     = np.array([[1] * BATCH_SIZE])
            labels_all     = np.concatenate((real_fake_label.T,y_onehot_all),axis=1)
            labels_all2    = np.concatenate((real_label.T,y_onehot),axis=1)
            #print(labels_all.shape)
            #print(labels_all)
            #print(labels_all2.shape)
            
            if epoch==0 and index==0:
                print("Input of G:", image_batch_c.shape)
                print("Input of D fake:", generated_images.shape)            
                print("Input of D real:", image_batch_lda.shape)            

            d_loss = discriminator.train_on_batch([real_pairs, fake_pairs],labels_all)

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)

            #print("%03d : batch %d d_loss : %f" % (epoch, index, d_loss))
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch([image_batch_c, noise, image_batch_lda], labels_all2)
            discriminator.trainable = True
            #print("%03d : batch %d g_loss : %f" % (epoch, index, g_loss))
            if 1==1:
                out_tr = discriminator.predict([image_batch_lda,generated_images])
                y_pr = np.argmax((out_tr[:,1:n_classes+1]), axis=1)
                train_cost_avg += 100 - 100 * accuracy_score(image_batch_y, y_pr)         
            
            if (index % int(X_train.shape[0]/BATCH_SIZE/10) == 0 and index>0) or index == int(X_train.shape[0]/BATCH_SIZE)-1:
            #if (index % 10 == 0 and index>0) or index == int(X_train.shape[0]/BATCH_SIZE)-1:
                loss_str = "epoch %03d batch %d: d_loss: %0.5f; dg_loss: %.5f" % (epoch, index, d_loss,g_loss)
                print(loss_str)
                sys.stdout.flush()

                ###Checking traing and test matching or not?
                print("Checking: max_train, min_train: %f, %f" % (np.max(generated_images), np.min(generated_images)))
                print("Checking: max_test, min_test: %f, %f" % (np.max(X_test_lda), np.min(X_test_lda)))
  
                # Calculate training accuracy
                loss_str="Cost on train: %.2f" % (train_cost_avg/(index+1))
                if 1==12:
                    noise1 = np.zeros((X_train.shape[0], 100))
                    for i in range(X_train.shape[0]):
                        noise1[i, :] = np.random.uniform(-1, 1, 100)
                    generated_images_test = generator.predict([X_train,noise1])
                    out_tr = discriminator.predict([X_train_lda,generated_images_test])
                    y_pr = np.argmax((out_tr[:,1:n_classes+1]), axis=1)
                    train_this_cost = 100 - 100 * accuracy_score(Y_train, y_pr)
                    loss_str = "Cost on train: %.2f" % (train_this_cost)
                
                # Calculate dev accuracy
                noise1 = np.zeros((X_dev.shape[0], 100))
                for i in range(X_dev.shape[0]):
                    noise1[i, :] = np.random.uniform(-1, 1, 100)
                generated_images_test = generator.predict([X_dev,noise1])
                out_tr = discriminator.predict([X_dev_lda,generated_images_test])
                print(out_tr.shape)
                print(out_tr[:,1:n_classes+1].shape)
                y_pr = np.argmax((out_tr[:,1:n_classes+1]), axis=1)
                dev_this_cost = 100 - 100 * accuracy_score(Y_dev, y_pr)
                print(y_pr.shape)
                loss_str1 = "; dev: %.2f" % (dev_this_cost)
                
                # Calculate test accuracy
                noise1 = np.zeros((X_test.shape[0], 100))
                for i in range(X_test.shape[0]):
                    noise1[i, :] = np.random.uniform(-1, 1, 100)               
                generated_images_test = generator.predict([X_test,noise1])
                out_tr = discriminator.predict([X_test_lda,generated_images_test])
                #out_tr = discriminator.predict([X_test_lda,X_test_lda])
                #out_tr = discriminator.predict([generated_images_test,generated_images_test])
                #out_tr = discriminator.predict([generated_images_test,X_test_lda])
                #print('X_test_lda.shape=',X_test_lda.shape)
                #print('generated_images_test.shape=',generated_images_test.shape)
                #print('out_tr.shape=',out_tr.shape)
                y_pr = np.argmax((out_tr[:,1:n_classes+1]), axis=1)
                #print('Y_test.shape=', Y_test.shape)
                #print('Y_test=',Y_test)
                #print('y_pr.shape=', y_pr.shape)
                #print('y_pr=', y_pr)

                test_this_cost = 100 - 100 * accuracy_score(Y_test, y_pr)
                loss_str2 = "; test: %.2f" % (test_this_cost)
                print(loss_str2)
                #exit()                
                ### Save model
                if dev_this_cost<dev_best_cost:
                    print(col.print_colored(loss_str+loss_str1+loss_str2, col.OKGREEN))
                    dev_best_cost = dev_this_cost
                
                    generator.save_weights(savePath+"/model/dev_best_generator", True)
                    discriminator.save_weights(savePath+"/model/dev_best_discriminator", True)
                else:
                    print(col.print_colored(loss_str+loss_str1+loss_str2, col.WARNING))
                    
                if test_this_cost<test_best_cost:
                    test_best_cost = test_this_cost
                
                    generator.save_weights(savePath+"/model/test_best_generator", True)
                    discriminator.save_weights(savePath+"/model/test_best_discriminator", True)
                
                if index == int(X_train.shape[0]/BATCH_SIZE)-1:
                    print(col.print_colored(loss_str+loss_str1+loss_str2, col.OKBLUE))
                    loss_str3 = "  Best Cost: dev %.2f, test %.2f %%" % (dev_best_cost, test_best_cost)
                    print(col.print_colored(loss_str3, col.OKBLUE))


def generateData_cGAN_classifierV01(BATCH_SIZE, savePath, Noise=True):
    print("##############Prepare to generate data with GAN")
    #np.set_printoptions(linewidth=500)
    #os.system("mkdir -p "+savePath+"/generate")
    import IS2017_model_02 as model
    model.feature_dim=49
    model.n_classes=50
    #pdb.set_trace()
    
    
    generator = model.generator_model()
    #epoch=90
    #index=50
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    #generator.load_weights(savePath+"/model/"+str(epoch)+"_"+str(index)+"_generator")
    generator.load_weights(savePath+"/model/dev_best_generator")
    
    data = load_LIDSet1All();
    X_train, Y_train = data['X_train'], data['y_train']
    X_dev, Y_dev = data['X_valid'], data['y_valid']
    X_test, Y_test = data['X_test'], data['y_test']
    

    
    X_train, Y_train=shuffle(X_train,Y_train)
    X_dev, Y_dev=shuffle(X_dev,Y_dev)
    X_test, Y_test=shuffle(X_test,Y_test)
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, Y_train)
    X_train=clf.transform(X_train)
    X_dev=clf.transform(X_dev)
    X_test=clf.transform(X_test)
    
    X_train_o = X_train
    X_dev_o = X_dev
    X_test_o = X_test    
    
    print(X_train.shape)
    print(X_dev.shape)
    print(X_test.shape)

    #pdb.set_trace()
    ### train  
    BATCH_SIZE = X_train.shape[0]
    #BATCH_SIZE = 128
    noise = np.zeros((BATCH_SIZE, 100))
    if Noise:
        for j in range(BATCH_SIZE):
            noise[j, :] = np.random.uniform(-1, 1, 100)
    X_train = generator.predict([X_train,noise], verbose=1)
    ### dev  
    BATCH_SIZE = X_dev.shape[0]
    #BATCH_SIZE = 128
    noise = np.zeros((BATCH_SIZE, 100))
    if Noise:
        for j in range(BATCH_SIZE):
            noise[j, :] = np.random.uniform(-1, 1, 100)
    X_dev = generator.predict([X_dev,noise], verbose=1)
    ### train  
    BATCH_SIZE = X_test.shape[0]
    #BATCH_SIZE = 128
    noise = np.zeros((BATCH_SIZE, 100))
    if Noise:
        for j in range(BATCH_SIZE):
            noise[j, :] = np.random.uniform(-1, 1, 100)
    X_test = generator.predict([X_test,noise], verbose=1)

    #min_max_scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    #for j in range(0,generated_images.shape[0]):
    #    generated_images[j][0] = min_max_scaler1.fit_transform(generated_images[j][0])       
    #print(generated_images[0][0])
        
    return dict(X_train=X_train,X_train_o=X_train_o, y_train=Y_train,
                X_valid=X_dev,X_valid_o=X_dev_o, y_valid=Y_dev,
                X_test=X_test,X_test_o=X_test_o, y_test=Y_test)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument("--savePath", type=str)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, savePath=args.savePath)
    elif args.mode == "generate":
        generateData_cGAN_classifierV01(BATCH_SIZE=args.batch_size, savePath=args.savePath)
