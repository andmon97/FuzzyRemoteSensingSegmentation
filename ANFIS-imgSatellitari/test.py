import torch
from utility import make_one_hot, num_cat_correct

def test_model(model, X_test, y_test, num_categories, k_fold, i, lista_acc):

    num_categories = num_categories
    y_test = make_one_hot(y_test, num_categories=num_categories)  # num_categories 3 iris 4 miracle
    x_Test_Tens = torch.Tensor(X_test)
    nc, tot = num_cat_correct(model, x_Test_Tens, y_test)

    if k_fold:
        print('RISULTATI TEST - ' + str(i))
        print('{} of {} correct (accuracy={:5.2f}%)'.format(nc, tot, nc * 100 / tot))
        acc = nc * 100 / tot
        lista_acc.append(acc)
        return lista_acc
    else:
        print('RISULTATI TEST')
        print('{} of {} correct (accuracy={:5.2f}%)'.format(nc, tot, nc * 100 / tot))