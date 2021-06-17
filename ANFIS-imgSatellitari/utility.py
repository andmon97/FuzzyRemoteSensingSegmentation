import matplotlib.pyplot as plt
import numpy as np
from membership import BellMembFunc, GaussMembFunc, TriangularMembFunc, TrapezoidalMembFunc
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE


def triang_to_gauss(mfs):
    """
    Transform membership function from Triangular into Gaussian function
    :param mfs: membership function
    :return: mfs_news
    """
    mfs_new = []
    for var in mfs:
        mu = var[1]
        sigma = np.abs(var[0]-var[2])/2
        sigma = round(np.sqrt(sigma), 0) #MOD SUGGERITA DALLA PROF.
        mfs_new.append((mu, sigma))
    return mfs_new

def split_dataset(dataset):
    """
    Split the dataset into data and terget
    :param dataset: dataset splitted
    :return: d_data, d_target
    """

    dataframe = pd.read_csv(dataset, header=None, sep=';')
    array = dataframe.values
    d_data = array[:, 0:len(dataframe.columns) - 1]
    d_target = array[:, len(dataframe.columns) - 1]

    return d_data, d_target

def load_dataset(dataset):
    """
    Load the features of dataset
    :param dataset:
    :return: d_data
    """
    dataframe = pd.read_csv(dataset, header=None, sep=';')
    array = dataframe.values
    d_data = array[:, 0:len(dataframe.columns)]
    return d_data

def load_target(dataset):
    """
    Load the target of dataset
    :param dataset:
    :return: arr
    """
    lines = open(dataset).read().split('\n')
    new_numbers = []
    for n in lines:
        new_numbers.append(int(n))
    arr = np.array(new_numbers)
    return arr


def load_model(name_variable, membership_function, mfs, n_terms):
    """
    Loading of input variables and definition of membership functions by ANFIS
    :param name_variable:
    :param membership_function:
    :param mfs:
    :param n_terms:
    :return: invardefs
    """

    invardefs = []

    if membership_function == BellMembFunc:
        print('Not implemented')
    elif membership_function == GaussMembFunc:
        n = len(name_variable)
        j = 0
        jj = n_terms
        for i in range(n):
            invardefs.append((name_variable[i], [(membership_function(*mfs)) for mfs in mfs[j:jj]][:n_terms]))
            j = jj
            jj = jj + n_terms
    elif membership_function == TriangularMembFunc:
        n = len(name_variable)
        j = 0
        jj = n_terms
        for i in range(n):
            invardefs.append((name_variable[i], [(membership_function(*mfs)) for mfs in mfs[j:jj]][:n_terms]))
            j = jj
            jj = jj + n_terms
    elif membership_function == TrapezoidalMembFunc:
        print('Not implemented')
    else:
        print('Not exist')

    return invardefs

class plot_import():
    def __init__(self, model, membership_function):
        """
        Variables initialisation
        :param model:
        :param membership_function:
        """
        self.model = model
        self.membership_function = membership_function

    def plot_import_mfs(self, k, fig):
        """
        :param k:
        :param fig:
        :return:
        """
        for i, (var_name, fv) in enumerate(self.model.layer.fuzzify.varmfs.items()):
            self._plot_import_mfs(var_name, fv, self.membership_function, k, fig)

    def _plot_import_mfs(self, var_name, fv, membership_function, k, fig):
        """
        Viewing graphs
        :param var_name:
        :param fv:
        :param membership_function:
        :param k:
        :param fig:
        :return:
        """
        self.membership_function = membership_function

        app = []
        mu_list = []
        sigma_list = []
        for mfname, mfdef in fv.mfdefs.items():
            i = 0
            for n, p in mfdef.named_parameters():
                app.append(p.data)
                if n == 'sigma':
                    sigma_list.append(p.data)
                else:
                    mu_list.append(p.data)
                i = i + 1

        if membership_function == BellMembFunc:
            print('Not implemented')
            exit()
        elif membership_function == GaussMembFunc:
            minimo = min(mu_list)
            massimo = max(mu_list)
            x = np.linspace(minimo, massimo, num=10000)
            x_1 = torch.tensor(x)
            i = 0
            while i < len(sigma_list):
                val = torch.exp(-torch.pow(x_1 - mu_list[i], 2) / (2 * sigma_list[i] ** 2))
                plt.plot(x_1.tolist(), val.tolist())
                i = i + 1
        elif membership_function == TriangularMembFunc:
            minimo = min(mu_list)
            massimo = max(mu_list)

            z = np.linspace(minimo.detach().numpy(), massimo.detach().numpy(), num=1000)
            i = 0
            j = 0
            while i < len(mfname):
                y = self._triangle(z, app[j].detach().numpy(), app[j + 1].detach().numpy(), app[j + 2].detach().numpy())
                plt.plot(z, y, label=mfname)
                j = j + 3
                i = i + 1
        elif membership_function == TrapezoidalMembFunc:
            print('Not implemented')
            exit()
        else:
            print('Not exist')
            exit()

        plt.xlabel('Values for variable {} ({} MFs)'.format(var_name, fv.num_mfs))
        plt.ylabel('Membership')
        plt.savefig('figure/'+ str(fig) + 'Fold_'+str(k)+'_'+str(var_name))
        plt.clf()
        #plt.show()

    def _triangle(self, z, a, b, c):
        """
        Triangular graph design
        :param z:
        :param a:
        :param b:
        :param c:
        :return: y
        """
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half] - a) / (b - a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c - z[second_half]) / (c - b)
        return y


def num_cat_correct(model, x, y_actual):
    """
    Work out the number of correct categorisations the model gives.
    Assumes the model is producing (float) scores for each category.
    Use a max function on predicted/actual to get the category.
    :param model:
    :param x:
    :param y_actual:
    :return: num_correct.item(), len(x)
    """

    y_pred = model(x)
    # Change the y-value scores back into 'best category':
    cat_act = torch.argmax(y_actual, dim=1)
    cat_pred = torch.argmax(y_pred, dim=1)
    num_correct = torch.sum(cat_act == cat_pred)
    return num_correct.item(), len(x)


def make_one_hot(data, num_categories, dtype=torch.float):
    """
     Take a list of categories and make them into one-hot vectors;
     that is, treat the original entries as vector indices.
     Return a tensor of 0/1 floats, of shape: len(data) * num_categories
    :param data:
    :param num_categories:
    :param dtype:
    :return: y
    """

    num_entries = len(data)
    # Convert data to a torch tensor of indices, with extra dimension:
    cats = torch.Tensor(data).long().unsqueeze(1)
    # Now convert this to one-hot representation:
    y = torch.zeros((num_entries, num_categories), dtype=dtype)\
        .scatter(1, cats, 1)
    y.requires_grad = True
    return y

def rules_parse(model, name_fuzzy_set, name_variable, model_l, membership_function):

    '''

    :param model:
    :param name_fuzzy_set:
    :param name_variable:
    :param model_l:
    :param membership_function:
    :return:
    '''

    lista_varname = []
    lista_mfname = []
    lista_mfdef = []
    lista_n = []
    lista_p = []
    app = []
    mu_list = []
    sigma_list = []
    a_list = []
    b_list = []
    c_list = []

    r = ['Input variables']

    i = 0
    for varname, members in model.layer['fuzzify'].varmfs.items():
            j = 0
            lista_varname.append(varname)
            for mfname, mfdef in members.mfdefs.items():
                lista_mfname.append(mfname)
                lista_mfdef.append(mfdef)
                if membership_function == BellMembFunc:
                    print('Not implemented')
                    exit()
                elif membership_function == GaussMembFunc:
                    for n, p in mfdef.named_parameters():
                        lista_n.append(n)
                        lista_p.append(p.data)
                        app.append(p.data)
                        if n == 'sigma':
                            sigma_list.append(p.data)
                        else:
                            mu_list.append(p.data)

                elif membership_function == TriangularMembFunc:
                    for n, p in mfdef.named_parameters():
                        lista_n.append(n)
                        lista_p.append(p.data)
                        app.append(p.data)
                        if n == 'a':
                            a_list.append(p.data)
                        elif n == 'b':
                            b_list.append(p.data)
                        else:
                            c_list.append(p.data)

                elif membership_function == TrapezoidalMembFunc:
                    print('Not implemented')
                    exit()

                else:
                    print('Not exist')
                    exit()

                r.append('- {}: {}({})'.format(varname, name_fuzzy_set[i][j],
                                               mfdef.__class__.__name__,
                                               ', '.join(['{}={}'.format(n, p.item())
                                                          for n, p in mfdef.named_parameters()])))

                '\n'.join(r)
                j = j+1
            i = i+1

    rstr = []
    vardefs = model.layer['fuzzify'].varmfs

    rule_ants = model.layer['rules'].extra_repr(vardefs).split('\n')
    x = model.layer['consequent'].coeff

    list_i = []
    list_rule_ants = []
    list_crow = []
    list_rule_conseguent = []


    scaler = MinMaxScaler()
    lis = []
    i = 0
    ##############################  NORMALIZZAZIONE #######################################
    y = Variable(x, requires_grad=True)
    y = y.detach().numpy()

    while i < len(y):
        scaled = scaler.fit_transform(y[i])
        lis.append(scaled)
        i = i + 1
    #######################################################################################

    list_prova = []
    for i, crow in enumerate(lis):
        list_i.append(i)
        list_rule_ants.append(rule_ants[i])
        for x in crow:
            list_prova.append(x.item())
        #print('*****************************')
        rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
        rstr.append(' ' * 9 + 'THEN {}'.format(crow.tolist()))

    i = 0
    j = 4
    while i < len(list_prova):
        list_crow.append(list_prova[i:j])
        i = j
        j = j + 4

    i = 0
    rule_not_active = []
    while i < len(list_crow):
        somma = np.sum(list_crow[i])
        if somma == 0:
            rule_not_active.append((i, list_crow[i]))
        i = i+1

    i = 0
    while i < len(list_crow):
        max = np.max(list_crow[i])
        max_index = list_crow[i].index(max)
        if max_index == 0:
            list_rule_conseguent.append('Risk_low')
        elif max_index == 1:
            list_rule_conseguent.append('Risk_medium')
        elif max_index == 2:
            list_rule_conseguent.append('Risk_high')
        elif max_index == 3:
            list_rule_conseguent.append('Risk_very_high')

        i = i + 1

    if model_l == False:
        i = 0
        while i < len(list_rule_ants):
            list_rule_ants[i] = list_rule_ants[i].replace(str(lista_varname[0]), str(name_variable[0])).replace(str(lista_varname[1]), str(name_variable[1])).replace(str(lista_varname[2]), str(name_variable[2])).replace(str(lista_varname[3]), str(name_variable[3]))
            i = i+1
        lista_varname = name_variable

    j = 0
    while j < len(lista_varname):
        z = 0
        while z < len(lista_mfname):
            i = 0
            while i < len(list_rule_ants):
                if str(lista_varname[j]) == 'HR':
                        if str(lista_mfname[z]) == 'mf0':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j])+' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j])+' IS ' + str(name_fuzzy_set[j][0]+')'))
                        elif str(lista_mfname[z]) == 'mf1':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j]) + ' is ' + str(lista_mfname[z]),'(' + str(lista_varname[j]) + ' IS ' + str(name_fuzzy_set[j][1]+')'))
                        elif str(lista_mfname[z]) == 'mf2':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j]) + ' is ' + str(lista_mfname[z]), '('+ str(lista_varname[j]) + ' IS ' + str(name_fuzzy_set[j][2]+')'))
                elif str(lista_varname[j]) == 'RR':
                        if str(lista_mfname[z]) == 'mf0':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j])+' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j])+' IS ' + str(name_fuzzy_set[j][0]+')'))
                        elif str(lista_mfname[z]) == 'mf1':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j]) + ' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j]) + ' IS ' + str(name_fuzzy_set[j][1]+')'))
                        elif str(lista_mfname[z]) == 'mf2':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j]) + ' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j]) + ' IS ' + str(name_fuzzy_set[j][2]+')'))
                elif str(lista_varname[j]) == 'SPO2':
                        if str(lista_mfname[z]) == 'mf0':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j])+' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j])+' IS ' + str(name_fuzzy_set[j][0]+')'))
                        elif str(lista_mfname[z]) == 'mf1':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j]) + ' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j]) + ' IS ' + str(name_fuzzy_set[j][1]+')'))
                        elif str(lista_mfname[z]) == 'mf2':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j]) + ' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j]) + ' IS ' + str(name_fuzzy_set[j][2]+')'))
                elif str(lista_varname[j]) == 'LIP':
                        if str(lista_mfname[z]) == 'mf0':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j])+' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j])+' IS ' + str(name_fuzzy_set[j][0]+')'))
                        elif str(lista_mfname[z]) == 'mf1':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j]) + ' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j]) + ' IS ' + str(name_fuzzy_set[j][1]+')'))
                        elif str(lista_mfname[z]) == 'mf2':
                            list_rule_ants[i] = str(list_rule_ants[i]).replace(str(lista_varname[j]) + ' is ' + str(lista_mfname[z]), '(' + str(lista_varname[j]) + ' IS ' + str(name_fuzzy_set[j][2]+')'))
                list_rule_ants[i] = str(list_rule_ants[i]).replace('and', 'AND')
                i = i + 1
            z = z + 1
        j = j+1

    return lista_varname, lista_mfname, name_fuzzy_set, lista_mfdef, sigma_list , mu_list, a_list, b_list, c_list,\
           list_rule_ants, list_rule_conseguent, rule_not_active


def fcl_write(lista_varname, lista_mfname, name_fuzzy_set, name_variable_output, outvars, lista_mfdef, sigma_list,
              mu_list, a_list, b_list, c_list, list_rule_ants, list_rule_conseguent, membership_function, i):
    '''

    :param lista_varname:
    :param lista_mfname:
    :param name_fuzzy_set:
    :param name_variable_output:
    :param outvars:
    :param lista_mfdef:
    :param sigma_list:
    :param mu_list:
    :param a_list:
    :param b_list:
    :param c_list:
    :param list_rule_ants:
    :param list_rule_conseguent:
    :param membership_function:
    :param i:
    :return:
    '''

    fcl = open("base"+str(i)+".fcl", "w")

    fcl.write('FUNCTION_BLOCK dummy \n')
    fcl.write('\n')
    fcl.write('	VAR_INPUT \n')

    if membership_function == GaussMembFunc:
        i = 0
        primo = 0
        ultimo = 3
        while i < len(lista_varname):
            minimo = min(mu_list[primo:ultimo])
            massimo = max(mu_list[primo:ultimo])

            fcl.write('		' + str(lista_varname[i]) + ' :	 REAL;(* RANGE(' + str(minimo.detach().numpy()) +' .. ' + str(massimo.detach().numpy()) +') *)' + '\n')

            primo = primo + len(name_fuzzy_set[i])
            ultimo = ultimo + len(name_fuzzy_set[i])

            i = i+1

    elif membership_function == TriangularMembFunc:
        i = 0
        primo = 0
        ultimo = 3
        while i < len(lista_varname):
            minimo = min(a_list[primo:ultimo])
            massimo = max(c_list[primo:ultimo])
            primo = primo + len(name_fuzzy_set[i])
            ultimo = ultimo + len(name_fuzzy_set[i])
            fcl.write('		' + str(lista_varname[i]) + ' :	 REAL; (* RANGE(' + str(minimo.detach().numpy()) +' .. '+ str(massimo.detach().numpy()) +') *)' + '\n')
            i = i + 1

    fcl.write('	END_VAR \n\n')

    fcl.write('	VAR_OUTPUT \n')
    fcl.write('		' + str(name_variable_output[0]) + ' : REAL; (* RANGE( .. ) *) \n')
    fcl.write('    END_VAR \n\n')


    if membership_function == GaussMembFunc:
        i = 0
        conta = 0
        while i < len(lista_varname):
            fcl.write('	FUZZIFY '+str(lista_varname[i])+'\n')
            j = 0
            while j < len(name_fuzzy_set[i]):
                mu = mu_list[conta]
                sigma = sigma_list[conta]
                fcl.write('		TERM '+str(name_fuzzy_set[i][j])+' := ('+str(mu.detach().numpy())+','+str(sigma.detach().numpy())+') ;\n')
                j = j+1
                conta = conta + 1
            fcl.write('	END_FUZZIFY \n\n')
            i = i+1


    elif membership_function == TriangularMembFunc:
        i = 0
        conta = 0
        while i < len(lista_varname):
            fcl.write('	FUZZIFY ' + str(lista_varname[i]) + '\n')

            j = 0
            while j < len(name_fuzzy_set[i]):
                a = a_list[conta]
                b = b_list[conta]
                c = c_list[conta]
                fcl.write('		TERM ' + str(name_fuzzy_set[i][j]) + ' := (' + str(a.detach().numpy()) + ', 0) ('
                          + str(b.detach().numpy())+', 1) (' + str(c.detach().numpy())+', 0)' + ';\n')

                j = j + 1
                conta = conta + 1
            fcl.write('	END_FUZZIFY \n\n')
            i = i + 1

    i = 0
    j = 1
    fcl.write('	DEFUZZIFY '+str(name_variable_output[0])+'\n')
    while i < len(outvars):


        fcl.write('		TERM ' + str(outvars[i]) +' := '+str(j)+';\n')

        j = j + 1
        i = i + 1
    fcl.write('		ACCU:MAX;\n')
    fcl.write('		METHOD:Dict;\n')
    fcl.write('		DEFAULT := 0;\n')
    fcl.write('	END_DEFUZZIFY\n\n')

    fcl.write('	RULEBLOCK first\n')
    fcl.write('		AND:MIN;\n')

    i = 0
    while i < len(list_rule_ants):
        fcl.write('		RULE '+str(i)+':IF '+str(list_rule_ants[i])+ ' THEN ('+ 'RISK_LEVEL IS ' +str(list_rule_conseguent[i]) +');'+'\n')

        i = i + 1

    fcl.write('	END_RULEBLOCK\n\n')

    fcl.write('END_FUNCTION_BLOCK \n')
    fcl.close()



def fcl_write_cut(lista_varname, lista_mfname, name_fuzzy_set, name_variable_output, outvars, lista_mfdef, sigma_list,
              mu_list, a_list, b_list, c_list, list_rule_ants, list_rule_conseguent, list_rule_cut, membership_function, i):
    '''

    :param lista_varname:
    :param lista_mfname:
    :param name_fuzzy_set:
    :param name_variable_output:
    :param outvars:
    :param lista_mfdef:
    :param sigma_list:
    :param mu_list:
    :param a_list:
    :param b_list:
    :param c_list:
    :param list_rule_ants:
    :param list_rule_conseguent:
    :param membership_function:
    :param i:
    :return:
    '''


    fcl_cut = open("base-cut"+str(i)+".fcl", "w")

    fcl_cut.write('FUNCTION_BLOCK dummy \n')
    fcl_cut.write('\n')
    fcl_cut.write('	VAR_INPUT \n')

    if membership_function == GaussMembFunc:
        i = 0
        primo = 0
        ultimo = 3
        while i < len(lista_varname):
            minimo = min(mu_list[primo:ultimo])
            massimo = max(mu_list[primo:ultimo])

            fcl_cut.write('		' + str(lista_varname[i]) + ' :	 REAL;(* RANGE(' + str(minimo.detach().numpy()) +' .. ' + str(massimo.detach().numpy()) +') *)' + '\n')

            primo = primo + len(name_fuzzy_set[i])
            ultimo = ultimo + len(name_fuzzy_set[i])

            i = i+1

    elif membership_function == TriangularMembFunc:
        i = 0
        primo = 0
        ultimo = 3
        while i < len(lista_varname):
            minimo = min(a_list[primo:ultimo])
            massimo = max(c_list[primo:ultimo])
            primo = primo + len(name_fuzzy_set[i])
            ultimo = ultimo + len(name_fuzzy_set[i])
            fcl_cut.write('		' + str(lista_varname[i]) + ' :	 REAL; (* RANGE(' + str(minimo.detach().numpy()) +' .. '+ str(massimo.detach().numpy()) +') *)' + '\n')
            i = i + 1

    fcl_cut.write('	END_VAR \n\n')

    fcl_cut.write('	VAR_OUTPUT \n')
    fcl_cut.write('		' + str(name_variable_output[0]) + ' : REAL; (* RANGE( .. ) *) \n')
    fcl_cut.write('    END_VAR \n\n')


    if membership_function == GaussMembFunc:
        i = 0
        conta = 0
        while i < len(lista_varname):
            fcl_cut.write('	FUZZIFY '+str(lista_varname[i])+'\n')
            j = 0
            while j < len(name_fuzzy_set[i]):
                mu = mu_list[conta]
                sigma = sigma_list[conta]
                fcl_cut.write('		TERM '+str(name_fuzzy_set[i][j])+' := ('+str(mu.detach().numpy())+','+str(sigma.detach().numpy())+') ;\n')
                j = j+1
                conta = conta + 1
            fcl_cut.write('	END_FUZZIFY \n\n')
            i = i+1


    elif membership_function == TriangularMembFunc:
        i = 0
        conta = 0
        while i < len(lista_varname):
            fcl_cut.write('	FUZZIFY ' + str(lista_varname[i]) + '\n')

            j = 0
            while j < len(name_fuzzy_set[i]):
                a = a_list[conta]
                b = b_list[conta]
                c = c_list[conta]
                fcl_cut.write('		TERM ' + str(name_fuzzy_set[i][j]) + ' := (' + str(a.detach().numpy()) + ', 0) ('
                          + str(b.detach().numpy())+', 1) (' + str(c.detach().numpy())+', 0)' + ';\n')

                j = j + 1
                conta = conta + 1
            fcl_cut.write('	END_FUZZIFY \n\n')
            i = i + 1

    i = 0
    j = 1
    fcl_cut.write('	DEFUZZIFY '+str(name_variable_output[0])+'\n')
    while i < len(outvars):


        fcl_cut.write('		TERM ' + str(outvars[i]) +' := '+str(j)+';\n')

        j = j + 1
        i = i + 1
    fcl_cut.write('		ACCU:MAX;\n')
    fcl_cut.write('		METHOD:Dict;\n')
    fcl_cut.write('		DEFAULT := 0;\n')
    fcl_cut.write('	END_DEFUZZIFY\n\n')

    fcl_cut.write('	RULEBLOCK first\n')
    fcl_cut.write('		AND:MIN;\n')

    i = 0
    while i < len(list_rule_cut):
        fcl_cut.write('		RULE '+str(i)+':IF '+str(list_rule_cut[i][1])+ ' THEN ('+ 'RISK_LEVEL IS ' +str(list_rule_cut[i][2]) +');'+'\n')

        i = i + 1

    fcl_cut.write('	END_RULEBLOCK\n\n')

    fcl_cut.write('END_FUNCTION_BLOCK \n')
    fcl_cut.close()



def rules_analysis(list_rule_ants, list_rule_conseguent, rule_not_active, i):
    '''
    :param list_rule_ants:
    :param list_rule_conseguent:
    :param rule_not_active:
    :param i:
    :return:
    '''


    k = i

    print('--------------------------------------------------------------------------------------')
    list_rule = []
    i = 0
    while i < len(list_rule_ants):
        list_rule.append((i, list_rule_ants[i], list_rule_conseguent[i]))
        i = i + 1

    i = 0
    conseguent_gold = []
    lista_str_to_int = []
    while i < len(list_rule):
        positive = list_rule[i][1].count("Normal")  #LABBRA ININFLUENTI
        positive_lip = list_rule[i][1].count("Regular")
        medium_lip = list_rule[i][1].count("Medium")
        irregular_lip = list_rule[i][1].count("Irregular")
        print('RULE'+str(i))
        print(positive)
        print(list_rule[i])
        if positive == 3:
            conseguent_gold.append('Risk_low')
            lista_str_to_int.append(0)
            print('Risk_low')
        elif positive == 2:
            conseguent_gold.append('Risk_medium')
            lista_str_to_int.append(1)
            print('Risk_medium')
        elif positive == 1:
            if irregular_lip == 1:
                conseguent_gold.append('Risk_very_high')
                lista_str_to_int.append(3)
                print('Risk_very_high')
            else:
                conseguent_gold.append('Risk_high')
                lista_str_to_int.append(2)
                print('Risk_high')
        elif positive == 0:
            conseguent_gold.append('Risk_very_high')
            lista_str_to_int.append(3)
            print('Risk_very_high')

        print('-----------------------------------------------------------------')
        i = i+1

    i = 0
    lista_controllo = []
    lista_equal_rules = []
    lista_NOT_equal_rules = []

    lista_str_to_int_2 = []
    i = 0
    while i<len(list_rule):
        if list_rule[i][2] == 'Risk_low':
            lista_str_to_int_2.append(0)
        if list_rule[i][2] == 'Risk_medium':
            lista_str_to_int_2.append(1)
        if list_rule[i][2] == 'Risk_high':
            lista_str_to_int_2.append(2)
        if list_rule[i][2] == 'Risk_very_high':
            lista_str_to_int_2.append(3)
        i = i+1

    i = 0
    while i < len(list_rule):
        if (conseguent_gold[i] == list_rule[i][2]):
            lista_equal_rules.append((list_rule[i][0], list_rule[i][1], list_rule[i][2]))
            lista_controllo.append(1)
        elif (conseguent_gold[i] != list_rule[i][2]):
            lista_NOT_equal_rules.append((list_rule[i][0], list_rule[i][1], list_rule[i][2]))
            lista_controllo.append(0)
        i = i+1


    lista_controllo_diff = []
    lista_controllo_diff_0 = []
    lista_controllo_diff_1 = []
    lista_controllo_diff_2 = []
    lista_controllo_diff_3 = []

    i = 0
    somma = 0
    while i < len(list_rule):
        diff = np.abs(lista_str_to_int[i] - lista_str_to_int_2[i])
        lista_controllo_diff.append(diff)
        if diff == 0:
            lista_controllo_diff_0.append((list_rule[i][0], list_rule[i][1], list_rule[i][2]))
        elif diff == 1:
            lista_controllo_diff_1.append((list_rule[i][0], list_rule[i][1], list_rule[i][2]))
        elif diff == 2:
            lista_controllo_diff_2.append((list_rule[i][0], list_rule[i][1], list_rule[i][2]))
        elif diff == 3:
            lista_controllo_diff_3.append((list_rule[i][0], list_rule[i][1], list_rule[i][2]))
        i = i + 1

    print('Analysis of rules \n')

    i = 0
    while i < len(rule_not_active):
        print('Rules not activated', rule_not_active[i][0])
        i = i+1

    print('\n')
    tot_define_rules = len(list_rule_conseguent) - len(rule_not_active)

    define_rules_w = []
    define_rules = []
    i = 0
    while i < len(lista_controllo_diff_0):
        j = 0
        flag = 1
        while j < len(rule_not_active):
            if lista_controllo_diff_0[i][0] == rule_not_active[j][0]:
                flag = 0
            j = j + 1
        if flag == 0:
            pass
        else:
            define_rules_w.append(lista_controllo_diff_0[i])
            define_rules.append(lista_controllo_diff_0[i])
        i = i + 1

    i = 0
    while i < len(define_rules):
        print(define_rules[i])
        i = i + 1
    print(len(define_rules), ' - fold Percentage of compared correct : ',
          round((len(define_rules) / tot_define_rules) * 100), '%')

    print('\n')
    define_rules = []
    i = 0
    while i < len(lista_controllo_diff_1):
        j = 0
        flag = 1
        while j < len(rule_not_active):
            if lista_controllo_diff_1[i][0] == rule_not_active[j][0]:
                flag = 0
            j = j + 1
        if flag == 0:
            pass
        else:
            define_rules_w.append(lista_controllo_diff_1[i])
            define_rules.append(lista_controllo_diff_1[i])
        i = i + 1

    i = 0
    while i < len(define_rules):
        print(define_rules[i])
        i = i + 1
    print(len(define_rules), ' - fold Percentage of compared not correct (class 1) : ',
          round((len(define_rules) / tot_define_rules) * 100), '%')

    print('\n')
    define_rules = []
    i = 0
    while i < len(lista_controllo_diff_2):
        j = 0
        flag = 1
        while j < len(rule_not_active):
            if lista_controllo_diff_2[i][0] == rule_not_active[j][0]:
                flag = 0
            j = j + 1
        if flag == 0:
            pass
        else:
            define_rules_w.append(lista_controllo_diff_2[i])
            define_rules.append(lista_controllo_diff_2[i])
        i = i+1

    i = 0
    while i < len(define_rules):
        print(define_rules[i])
        i = i+1
    print(str(len(define_rules)) + ' - fold Percentage of compared not correct (class 2) : ',
          round((len(define_rules) / tot_define_rules) * 100),'%')

    print('\n')

    define_rules = []
    i = 0
    while i < len(lista_controllo_diff_3):
        j = 0
        flag = 1
        while j < len(rule_not_active):
            if lista_controllo_diff_3[i][0] == rule_not_active[j][0]:
                flag = 0
            j = j + 1
        if flag == 0:
            pass
        else:
            define_rules_w.append(lista_controllo_diff_3[i])
            define_rules.append(lista_controllo_diff_3[i])
        i = i+1

    i = 0
    while i < len(define_rules):
        print(define_rules[i])
        i = i+1
    print(str(len(define_rules)) + ' - fold Percentage of compared not correct (class 3) : ',
          round((len(define_rules) / tot_define_rules) * 100),'%')


    i = 0
    list_rule_cut = []
    while i < len(list_rule):
        j = 0
        flag = 1
        while j < len(rule_not_active):
            if list_rule[i][0] == rule_not_active[j][0]:
                flag = 0
            j = j + 1
        if flag == 0:
            pass
        else:
            list_rule_cut.append(list_rule[i])
        i = i+1


    if len(define_rules_w) == len(list_rule_conseguent):
        list_rule_cut = None
        return list_rule_cut
    else:
        return list_rule_cut


def oversampling(dataset, plot=True):
    '''
    :param dataset:
    :param plot:
    :return:
    '''

    # load the csv file as a data frame
    df = pd.read_csv(dataset, header=None)
    data = df.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    # label encode the target variable
    y = LabelEncoder().fit_transform(y)
    # summarize distribution
    counter = Counter(y)
    print("Current class distribution")
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    # plot the distribution
    if plot == True:
        plt.title("Current class distribution")

        class_ = ['risk_low', 'risk_medium', 'risk_high', 'risk_very_high']
        x = np.array([0, 1, 2, 3])
        plt.xticks(x, class_)
        plt.bar(counter.keys(), counter.values())
        plt.show()

    # load the csv file as a data frame
    df = pd.read_csv(dataset, header=None)
    data = df.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    # label encode the target variable
    y = LabelEncoder().fit_transform(y)
    # transform the dataset
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    # summarize distribution
    counter = Counter(y)
    print('\nDistribution after SMOTE method')
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

    if plot == True:
        # plot the distribution
        plt.title("Class distribution after SMOTE method")
        class_ = ['risk_low', 'risk_medium', 'risk_high', 'risk_very_high']
        x = np.array([0, 1, 2, 3])
        plt.xticks(x, class_)
        plt.bar(counter.keys(), counter.values())
        plt.show()

    print(X)
    print(y)

    '''
    with open('datasets/miracle_X_over.csv', 'w') as FOUT:
        np.savetxt(FOUT, X, delimiter=',')

    with open('datasets/miracle_y_over.csv', 'w') as FOUT:
        np.savetxt(FOUT, y)
    '''


#################################   DEPRECATE  ######################################################
class plot_generate():

    def __init__(self, model, x, membership_function):
        """
        Variables initialisation
        :param model:
        :param x:
        :param membership_function:
        """
        self.model = model
        self.x = x
        self.membership_function = membership_function

    def plot_all_mfs(self):
        """
        :return:
        """
        for i, (var_name, fv) in enumerate(self.model.layer.fuzzify.varmfs.items()):
            self._plot_mfs(var_name, fv, self.x[:, i], self.membership_function)

    def _plot_mfs(self, var_name, fv, x, membership_function):
        """
        Viewing graphs
        :param var_name:
        :param fv:
        :param x:
        :param membership_function:
        :return:
        """

        if membership_function == BellMembFunc:
            print('Not implemented')
            exit()
        elif membership_function == GaussMembFunc:
            # Sort x so we only plot each x-value once:
            xsort, _ = x.sort()
            for mfname, yvals in fv.fuzzify(xsort):
                plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
        elif membership_function == TriangularMembFunc:
            xsort, _ = x.sort()
            for mfname, yvals in fv.fuzzify(xsort):
                plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
        elif membership_function == TrapezoidalMembFunc:
            print('Not implemented')
            exit()
        else:
            print('Not exist')
            exit()

        plt.xlabel('Values for variable {} ({} MFs)'.format(var_name, fv.num_mfs))
        plt.ylabel('Membership')
        plt.legend(bbox_to_anchor=(1., 0.95))
        plt.show()

#####################################################################################################


