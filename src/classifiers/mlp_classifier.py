# This script uses the high frequency features extracted 
# using matlab script provided by Dzanic et. al (https://github.com/tarikdzanic/FourierSpectrumDiscrepancies)
# and use a 2-layer MLP classifier to perform binary classification on real vs fake. 
# We use sigmoid activation functions

# General modules
import os, math

# Scientific computing libraries
import numpy as np

# Scikit-learn/ scipy libraries
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import metrics
from scipy.io import loadmat

# Other modules
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm



def load_mat_training_data( real_fts_dir: str, 
                            gan_fts_dir: str, 
                            num_examples: int, split: float):
    """""
    Load data from .mat files 
    
    Args:
        real_fts_dir    : Directory containing .mat files for real images
        gan_fts_dir     : Directory containing .mat files for GAN generated images
        num_examples    : number of samples for real / gan images
        split           : train/ val split (Between 0.0 and 1.0)
    """

    # Read files
    real_fts_files = [ os.path.join(real_fts_dir, i) for i in os.listdir(real_fts_dir) if i.endswith('.mat')]
    gan_fts_files = [ os.path.join(gan_fts_dir, i) for i in os.listdir(gan_fts_dir) if i.endswith('.mat')]

    # Sort files
    real_fts_files.sort()
    gan_fts_files.sort()

    # Choose random samples to train
    indexes = np.random.randint(len(real_fts_dir), size=num_examples)
    real_fts_files = [real_fts_files[i] for i in indexes]
    gan_fts_files = [gan_fts_files[i] for i in indexes]

    # Now split the files to train and test
    real_split_index = int( math.ceil(len(real_fts_files)*split) )
    gan_split_index = int ( math.ceil(len(gan_fts_files)*split) )
    
    # Define variables to acuumulate samples
    X_train, Y_train = [], []
    X_test_real, Y_test_real = [], []
    X_test_gan, Y_test_gan = [], []


    # Label for real=1. Get data for both train and test
    for i in real_fts_files[:real_split_index]:
        fts = list(loadmat(i)['c'][0])
        X_train.append(fts)
        Y_train.append(1)
    
    for i in real_fts_files[real_split_index:]:
        fts = list(loadmat(i)['c'][0])
        X_test_real.append(fts)
        Y_test_real.append(1)

    # Record this breakpoint if evaluation on real and fake 'training' data is required
    tr_real_data_break_point = len(X_train)

    # Label for fake=1, Get data for both train and test
    for i in gan_fts_files[:gan_split_index]:
        fts = list(loadmat(i)['c'][0])
        X_train.append(fts)
        Y_train.append(0)

    for i in gan_fts_files[gan_split_index:]:
        fts = list(loadmat(i)['c'][0])
        X_test_gan.append(fts)
        Y_test_gan.append(0)

    return X_train, Y_train, \
            X_test_real, Y_test_real,\
            X_test_gan, Y_test_gan, tr_real_data_break_point


def load_mat_testing_data(gan_fts_dir, num_examples):
    """""
    Load data from .mat files for testing
    
    Args:
        gan_fts_dir     : Directory containing .mat files for GAN generated images
        num_examples    : number of samples for GAN
    """
    gan_fts_files = [ os.path.join(gan_fts_dir, i) for i in os.listdir(gan_fts_dir) if i.endswith('.mat') ]
    gan_fts_files.sort()
    gan_fts_files = gan_fts_files[:num_examples]

    X_test, Y_test = [], []

    # Label for fake=1
    for i in gan_fts_files:
        fts = list(loadmat(i)['c'][0])
        X_test.append(fts)
        Y_test.append(0)

    return X_test, Y_test



def train_one_mlp_and_test( dataset,
                            adversarial_mode,
                            num_samples, split):
    """
    Train 1 MLP classifier (k=5) and test

    Args:
        dataset             : Used to identify path of features
        adversarial_mode    : Used to identify GAN path of features
        num_examples        : number of samples for real / gan images
        split               : train/ val split (Between 0.0 and 1.0)
    """
    # Load training and testing data for the model
    X_train, Y_train,\
    X_test_real, Y_test_real,\
    X_test_gan, Y_test_gan, \
    tr_real_data_break_point = load_mat_training_data('fits/{}/real/'.format(dataset), \
                                                 'fits/{}/{}/BASELINE/'.format(dataset, adversarial_mode), \
                                                 num_samples,\
                                                 split)
    
    print("Training data = {}, Real_testing_data = {}, GAN_testing data = {}".format(\
            len(X_train), len(X_test_real), len(X_test_gan)))

    # Defind a SVM
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', 
            learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, 
            tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
            beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
    print(mlp)

    # Preprocessing. Get scaler to normalize data
    scaler = preprocessing.StandardScaler().fit(X_train)

    # Fit on training data
    mlp.fit(scaler.transform(X_train), Y_train)

    # Get training accuracy on real and GAN data seperately
    train_acc_real, train_acc_fake = mlp.score(scaler.transform(X_train[:tr_real_data_break_point]), Y_train[:tr_real_data_break_point]), \
                                        mlp.score(scaler.transform(X_train[tr_real_data_break_point:]), Y_train[tr_real_data_break_point:])
    # train_acc_real = metrics.accuracy_score(pred_train_real, Y_train[:tr_real_data_break_point])
    # train_acc_fake = metrics.accuracy_score(pred_train_fake, Y_train[tr_real_data_break_point:])
    print("Training acc: real = {:.3f}, fake = {:.3f}".format(train_acc_real, train_acc_fake))

    # Get testing accuracy on real and GAN data seperately
    test_acc_real, test_acc_fake = mlp.score(scaler.transform(X_test_real), Y_test_real), \
                                        mlp.score(scaler.transform(X_test_gan), Y_test_gan)
    # test_acc_real = metrics.accuracy_score(pred_test_real, Y_test_real)
    # test_acc_fake = metrics.accuracy_score(pred_test_fake, Y_test_gan)
    print("Test acc: real = {:.3f}, fake = {:.3f}".format(test_acc_real, test_acc_fake) )

    final_accs = [train_acc_real, train_acc_fake, test_acc_real, test_acc_fake]
    setups = ['fits/{}/{}/{}/'.format(dataset, adversarial_mode, i) for i in \
                ['N.1.5', 'Z.1.5', 'B.1.5', \
                'N.1.3', 'N.1.7', \
                'Z.1.3', 'Z.1.7', \
                'B.1.3', 'B.1.7', \
                'N.3.5', 'Z.3.5', 'B.3.5' ]  ]

    # Test on other models iteratively
    for i in setups:
        X_test, Y_test = load_mat_testing_data(i, num_samples)
        #pred = svm_.predict(scaler.transform(X_test))
        acc = mlp.score(scaler.transform(X_test), Y_test)
        final_accs.append(acc)
        print("Accuracy for setup {} = {:3f}".format(i, acc))

    return final_accs


def train_multiple_mlps(model_str, num_experiments, num_samples, split):
    """
    Train multiple MLP classifier (k=5) and test. ALl results are recorded in a csv file
    
    Args:
        model_str           : Used to identify GAN 
        num_experiments     : Number of independent experiments to run (We run 10 experiments and average for the paper)
        num_samples         : number of samples for real / gan images
        split               : train/ val split (Between 0.0 and 1.0)
    """
    # Create column names
    exp_names = [ "train_acc_real", "train_acc_fake", "test_acc_real", "test_acc_fake" ]
    exp_names.extend( ['N.1.5', 'Z.1.5', 'B.1.5', \
        'N.1.3', 'N.1.7', \
        'Z.1.3', 'Z.1.7', \
        'B.1.3', 'B.1.7', \
        'N.3.5', 'Z.3.5', 'B.3.5' ] )

    # Create dataframe
    name = "{}/{}_training_examples/MLP".format(model_str, int(num_samples*split))
    df = pd.DataFrame(columns=[name])
    df[name] = exp_names
    
    # Run every experiment and update the dataframe
    for exp in tqdm(range(num_experiments)):
        final_accs = train_one_mlp_and_test('celeba', model_str, num_samples, split)
        df['exp{}'.format(exp)] = final_accs
        
    df.to_csv('csv/{}_mlp_results_{}_{}.csv'.format(model_str, num_samples, split), index=None)
    return


def main():
    # > Setup command line arguments
    parser = argparse.ArgumentParser()

    # Classifier arguments
    parser.add_argument('--model_str', required=True, choices=['gan', 'lsgan', 'wgan'])
    parser.add_argument('--num_exp', required=True, type=int)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--split', default=0.10, type=float)
    args = parser.parse_args()
    
    # Train KNN and save results
    train_multiple_mlps(args.model_str, args.num_exp, args.num_samples, args.split)



if __name__ == "__main__":
    # Place all the fits as follows:
    # fits/
    #   - celeba/
    #       - real/
    #       - gan/
    #           - BASELINE/
    #           - N.1.5/
    #           - Z.1.5/
    #           - ..../
    #       - lsgan/
    #           - BASELINE/
    #           - N.1.5/
    #           - Z.1.5/
    #           - ..../
    #       - wgan/
    #           - BASELINE/
    #           - N.1.5/
    #           - Z.1.5/
    #           - ..../
    
    main()