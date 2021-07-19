import numpy as np
import torch
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from membership import make_anfis, make_anfis_T
from utility import make_one_hot, plot_import
from membership import BellMembFunc, GaussMembFunc, TriangularMembFunc, TrapezoidalMembFunc



class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        """
        Variables initialisation
        :param X_data:
        :param y_data:
        """
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        """

        :param index:
        :return: self.X_data[index], self.y_data[index]
        """
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        """

        :return: len(self.X_data)
        """
        return len(self.X_data)

device = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
print(device)
print("Begin training.")

def multi_acc(y_pred, y_test):
    """
    Calculating accuracy
    :param y_pred:
    :param y_test:
    :return: acc
    """
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    #acc = torch.round(acc) * 100
    return acc


def get_data_one_hot(X_train, y_train, X_val, y_val, batch_size):
    """
    Return the iris data as a torch DataLoader object.
    There are n input features, but you can select fewer.
    The y-values are a one-hot representation of the categories.
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param batch_size:
    :return: DataLoader(train_dataset, batch_size=batch_size), DataLoader(val_dataset, batch_size=batch_size),\
               DataLoader(td, batch_size=batch_size, shuffle=False)
    """


    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    x = torch.Tensor(X_train)
    y = make_one_hot(y_train, num_categories=6)
    td = TensorDataset(x, y)

    if len(X_val) == 0:
        return DataLoader(train_dataset, batch_size=batch_size), DataLoader(td, batch_size=batch_size, shuffle=False)
    else:
        val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        return DataLoader(train_dataset, batch_size=batch_size), DataLoader(val_dataset, batch_size=batch_size),\
               DataLoader(td, batch_size=batch_size, shuffle=False)


def train_model(model, X_train, y_train, X_val, y_val, n_terms, num_categories, batch_size, epoch,
                model_l, hybrid, membership_function, i):
    """
    Preparation of Training a hybrid and not-hybrid Anfis based on the data.
    :param model:
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param n_terms:
    :param num_categories:
    :param batch_size:
    :param epoch:
    :param model_l:
    :param hybrid:
    :param membership_function:
    :param i:
    :return: model
    """

    train_data, val_data, x_train = get_data_one_hot(X_train, y_train, X_val, y_val, batch_size)
    x_train, y_train = x_train.dataset.tensors

    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99) Non FUNZIONA
    #optimizer = torch.optim.Rprop(model.parameters(), lr=1e-4)



    fig = 'pre-training'
    if model_l == False:
        if membership_function == BellMembFunc:
            print('Not implemented')
        elif membership_function == GaussMembFunc:
            model = make_anfis(x_train, num_mfs=n_terms, num_out=num_categories, hybrid=hybrid)
        elif membership_function == TriangularMembFunc:
            model = make_anfis_T(x_train, num_mfs=n_terms, num_out=num_categories, hybrid=hybrid)
        elif membership_function == TrapezoidalMembFunc:
            print('Not implemented')

        print('Model - Pre-training')
        print(model)

        plot_import(model, membership_function=membership_function).plot_import_mfs(i, fig)
    else:
        print('Model - Pre-training')
        print(model)
        plot_import(model, membership_function=membership_function).plot_import_mfs(i, fig)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) #lr=1e-6 lr=1e-4
    model = _train_anfis_cat(model, train_data, val_data, num_categories, optimizer, epoch, i)

    print('Model - Post-training', model)

    fig = 'post-training'
    plot_import(model, membership_function=membership_function).plot_import_mfs(i, fig)

    return model


def _train_anfis_cat(model, train_loader, val_loader, num_categories,optimizer, EPOCHS, i):
    """
    Training phase
    :param model:
    :param train_loader:
    :param val_loader:
    :param num_categories:
    :param optimizer:
    :param EPOCHS:
    :param i:
    :return: best_model
    """

    accuracy_stats = {'train': [], 'val': []}
    loss_stats = {'train': [], 'val': []}
    best_loss = np.inf
    best_epoch = 0

    for e in tqdm(range(1, EPOCHS + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0
            model.fit_coeff(X_train_batch.float(), make_one_hot(y_train_batch.float(), num_categories=num_categories))
            #model.eval()

            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        if (val_epoch_loss / len(val_loader)) < best_loss:
            best_loss = (val_epoch_loss / len(val_loader))
            best_epoch = e
            best_model = model
            print('best epoch',best_epoch)
            torch.save(model, 'models/'+str(i)+'G_model_geo' + str(best_epoch) + '.h5')
            # best_acc_epoch_pair[0] = epoch_acc
            # best_acc_epoch_pair[1] = e
            # early stopping
        if e - best_epoch > 20:
            print(best_epoch)
            break

        print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.8f} | Val Loss: {val_epoch_loss / len(val_loader):.8f} | Train Acc: {train_epoch_acc / len(train_loader):.8f}| Val Acc: {val_epoch_acc / len(val_loader):.8f}')

    plt.plot(loss_stats['train'], color='orange', label='Training')
    plt.plot(loss_stats['val'], color='blue', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('figure/Fold_' + str(i) + '_Val_loss')
    plt.clf()
    #plt.show()
    return best_model

################################## DEPRECATE ####################################################

'''
def train_model_kfold(model, X_train, y_train, X_val, y_val, n_terms, num_categories, batch_size, epoch,
                model_l, hybrid, membership_function, i):
    
        #Train a hybrid Anfis based on the Iris data.
        #I use a 'resilient' BP optimiser here, as SGD was a little flakey.
    

    train_data, x_train = get_data_one_hot(X_train, y_train, X_val, y_val, batch_size)

    x_train, y_train = x_train.dataset.tensors

    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99) Non FUNZIONA
    #optimizer = torch.optim.Rprop(model.parameters(), lr=1e-4)

    print('Model - Pre-training')
    print(model)

    ###############################PLOT PRIMA DEL TRAINING################################
    plot_generate(model, x_train, membership_function=membership_function).plot_all_mfs()
    #######################################################################################


    if model_l == False:

        model = make_anfis(x_train, num_mfs=n_terms, num_out=num_categories, hybrid=hybrid)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #lr=1e-6 1e-2
    model = _train_anfis_kfold_cat(model, train_data, num_categories, optimizer, epoch, i)


    ###########################USO STESSA FUNZIONE DI PLOT###################################
    plot_generate(model, x_train, membership_function=membership_function).plot_all_mfs()
    #########################################################################################
    
    if model_l == False:
        plot_generate(model, x_train, membership_function=membership_function).plot_all_mfs()
    else:
       plot_import(model, membership_function=membership_function).plot_import_mfs()
    
    return model

def _train_anfis_kfold_cat(model, train_data, num_categories, optimizer, epoch):
    accuracy_stats = {'train': [], 'val': []}
    loss_stats = {'train': [], 'val': []}
    best_accuracy = 0
    for e in tqdm(range(1, epoch + 1)):
        # Set current loss value
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        # Iterate over the DataLoader for training data
        # for i, data in enumerate(train_loader, 0):

        for X_train_batch, y_train_batch in train_data:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            y_train_pred = model(X_train_batch)

            # Compute loss
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            # Perform backward pass
            train_loss.backward()

            # Perform optimization
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        with torch.no_grad():
            model.fit_coeff(X_train_batch.float(), make_one_hot(y_train_batch.float(), num_categories=num_categories))

        loss_stats['train'].append(train_epoch_loss / len(train_data))
        accuracy_stats['train'].append(train_epoch_acc / len(train_data))

        cur_accuracy = train_epoch_acc / len(train_data)
        if cur_accuracy > best_accuracy:
            best_model = model
            best_epoch = e
            best_accuracy = cur_accuracy
            # best_acc_epoch_pair[0] = epoch_acc
            # best_acc_epoch_pair[1] = e
            # early stopping
        if e - best_epoch > 20:
            print('best epoch', best_epoch)
            break

        # Process is complete.
        print('Training process has finished. Saving trained model.')
        print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_data):.8f} | '
              f'Train Acc: {train_epoch_acc / len(train_data):.8f}')

    plt.plot(loss_stats['train'], color="orange")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return best_model



################################################################################################
'''