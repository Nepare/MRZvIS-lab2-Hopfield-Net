import ast
import numpy as np
import math
import random


MODEL_SIZE = 5


class HopfieldNetwork:
    remembered_models = []
    weight_matrix = []
    current_model = []

    def __init__(self):
        for weight_matrix_line in range(MODEL_SIZE ** 2):
            self.weight_matrix.append([])
            for weight_matrix_col in range(MODEL_SIZE ** 2):
                self.weight_matrix[weight_matrix_line].append(0)
        self.shuffled_neurons = []
        for shuffle_index in range(MODEL_SIZE ** 2):
            self.shuffled_neurons.append(shuffle_index)
        random.shuffle(self.shuffled_neurons)

    def remember_model(self, model):
        print(print_model(model))
        self.remembered_models.append(model)
        addition_to_weight_matrix = np.matmul(np.array([model]).T, np.array([model]))
        numpy_array_of_new_wm = (np.array(self.weight_matrix) + addition_to_weight_matrix)
        self.weight_matrix = numpy_array_of_new_wm.tolist()
        for nullifying_index in range(len(self.weight_matrix)):
            self.weight_matrix[nullifying_index][nullifying_index] = 0

    def recognize_model(self, model, async_mode_in):
        print(print_model(model) + "\n")
        counter = 0
        np_wm = np.array(self.weight_matrix)
        np_dist_model = np.array([model]).T
        relaxed = False
        next_state = np.array([0])

        asynch_mode = async_mode_in

        while not relaxed:
            counter += 1
            if counter > 10000:
                print("Невозможно опознать образ!\n")
                exit()
            previous_iteration = next_state

            multiplied_matrix = np.matmul(np_wm, np_dist_model)

            # =-=-=-= синхронный режим =-=-=-=

            if not asynch_mode:
                next_state = multiplied_matrix

                for activating_index in range(len(next_state)):
                    if next_state[activating_index][0] >= 0:
                        next_state[activating_index][0] = 1
                    else:
                        next_state[activating_index][0] = -1

            # =-=-=-=                  =-=-=-=

            # =-=-=-= асинхронный режим =-=-=-=

            if asynch_mode:
                if counter == 1:
                    next_state = np.array([model]).T

                cycle = counter % (MODEL_SIZE ** 2)
                current_random_neuron = self.shuffled_neurons[cycle]
                next_state[current_random_neuron][0] = multiplied_matrix[current_random_neuron][0]

                if next_state[current_random_neuron][0] >= 0:
                    next_state[current_random_neuron][0] = 1
                else:
                    next_state[current_random_neuron][0] = -1

            # =-=-=-=                   =-=-=-=

            np_dist_model = next_state
            print(print_model(next_state.T.flatten().tolist()) + "\n")

            if (previous_iteration == next_state).all():
                relaxed = True

    def return_weight_matrix(self):
        return self.weight_matrix


def print_model(model):
    image = ""
    for current_pixel in range(len(model)):
        if model[current_pixel] == 1:
            image = image + "@"
        if model[current_pixel] == -1:
            image = image + " "
        if (current_pixel + 1) % MODEL_SIZE == 0:
            image = image + "\n"
    return image


def convert_raw_text_into_matrix_of_models(raw_text):
    current_index_of_model_start = 0
    distinct_models = []
    for current_line in range(len(raw_text)):
        if raw_text[current_line] == '\n':
            new_model = raw_text[current_index_of_model_start:current_line]
            distinct_models.append(new_model)
            current_index_of_model_start = current_line + 1

    for model_index in range(len(distinct_models)):
        for line_index in range(len(distinct_models[0])):
            distinct_models[model_index][line_index] = distinct_models[model_index][line_index][:-1]
            vector_of_values = []
            for symbol in distinct_models[model_index][line_index]:
                if symbol == '1':
                    vector_of_values.append(1)
                if symbol == '0':
                    vector_of_values.append(-1)
            distinct_models[model_index][line_index] = vector_of_values

    flattened_distinct_models = []
    for model_index in range(len(distinct_models)):
        flattened_model = []
        for line_index in range(len(distinct_models[0])):
            flattened_model += distinct_models[model_index][line_index]
        flattened_distinct_models.append(flattened_model)
    return flattened_distinct_models


initial_choice = input("Введите режим работы программы: \n1 - обучение по эталонам\nВсё остальное - распознавание\n\n")
if initial_choice == '1':
    model_file_name = input("Введите название файла с эталонами: ")
    if model_file_name[-4:] != ".txt":
        model_file_name = model_file_name + ".txt"
    with open(model_file_name, "r") as model_file:
        models = model_file.readlines()
    distinct_models_list = convert_raw_text_into_matrix_of_models(models)

    net = HopfieldNetwork()
    for index_of_model in range(len(distinct_models_list)):
        net.remember_model(distinct_models_list[index_of_model])

    save_input = input("Образы запомнены!\nЕсли не хотите сохранять матрицу весов, нажмите Enter\n"
                       "Если хотите, введите название соответствующего файла\n\n")
    if save_input == "":
        pass
    else:
        if save_input[-4:] != ".txt":
            save_input = save_input + ".txt"
        wm_file = open(save_input, "w")
        wm_file.write(str(net.return_weight_matrix()))

else:
    wm_file_name = input("Введите название файла с обученной матрицей весов: ")
    if wm_file_name[-4:] != ".txt":
        wm_file_name = wm_file_name + ".txt"
    with open(wm_file_name, "r") as wm_file:
        weights = ast.literal_eval(wm_file.readline())

    distorted_model_name = input("Введите название файла с искажённым образом: ")
    if distorted_model_name[-4:] != ".txt":
        distorted_model_name = distorted_model_name + ".txt"
    with open(distorted_model_name, "r") as model_file:
        distorted_model = model_file.readlines()
    distinct_distorted_model = convert_raw_text_into_matrix_of_models(distorted_model)

    net = HopfieldNetwork()
    net.weight_matrix = weights

    s_mode: bool
    synch_mode = input("Введите 1 для синхронного режима\nВведите всё остальное для асинхронного режима\n\n")
    if synch_mode == "1":
        s_mode = False
    else:
        s_mode = True

    for current_model_index in range(len(distinct_distorted_model)):
        net.recognize_model(distinct_distorted_model[current_model_index], s_mode)
        print("Конец распознания!\n\n")

    # print(distinct_distorted_model)
