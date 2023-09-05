import os


def find_init_weight(dataset_txt_path: str, class_num: int, feature_num: int,
                     normalize: bool) -> (str, str):
    train_data = []
    weight_2d_list = []

    # inti dictionary
    result = {i: {'max': [-float('inf')] * feature_num, 'min': [float('inf')] * feature_num} for i in range(class_num)}

    # read data
    with open(dataset_txt_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # split by ' '
            data = list(map(float, line.strip().split(' ')))
            if normalize:
                nor_data = min_max_normalize(data[:feature_num])
                nor_data.append(int(data[-1]))
                data = nor_data
                train_data.append(data)
            else:
                data_0 = data[:feature_num]
                data_0.append(int(data[-1]))
                train_data.append(data_0)
            # get data's own type
            label = int(data[-1])
            # renew dictionary
            for i in range(feature_num):
                result[label]['max'][i] = max(result[label]['max'][i], data[i])
                result[label]['min'][i] = min(result[label]['min'][i], data[i])

    # output
    for i in range(class_num):
        print("Class ", i)
        print("Max values:", result[i]['max'])
        print("Min values:", result[i]['min'])
        weight_data = []
        for r in range(len(result[i]['max'])):
            weight_data.append(result[i]['min'][r])
            weight_data.append(result[i]['max'][r])
        weight_2d_list.append(weight_data)
    dataset_file_name = dataset_txt_path.split('/')[-1].split('.')[0]
    folder_path = os.path.dirname(dataset_txt_path)

    if normalize:
        output_weight_name = dataset_file_name + "_weight_normalized.txt"
        output_dataset_name = dataset_file_name + "_normalized.txt"
    else:
        output_weight_name = dataset_file_name + "_weight_not_normalized.txt"
        output_dataset_name = dataset_file_name + "_not_normalized.txt"
    output_weight_path = os.path.join(folder_path, output_weight_name)
    output_train_dataset_path = os.path.join(folder_path, output_dataset_name)

    write_in_txt_file(output_weight_path, weight_2d_list)
    write_in_txt_file(output_train_dataset_path, train_data)

    return output_train_dataset_path, output_weight_path


def write_in_txt_file(file_path: str, input_data: str) -> None:
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            print(f"build file successfully: {file_path}")
            for input_list in input_data:
                s = ' '.join(map(str, input_list))
                # s = s + " " + str(self.tactics)
                f.write('{}\n'.format(s))
    else:
        with open(file_path, 'w') as f:
            print(f"file exist, overwrite: {file_path}")
            for input_list in input_data:
                s = ' '.join(map(str, input_list))
                f.write('{}\n'.format(s))


def min_max_normalize(data: list) -> list:
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(val - min_val) / (max_val - min_val) for val in data]
    normalized_data = [round(val, 3) for val in normalized_data]
    return normalized_data


if __name__ == "__main__":
    _, _ = find_init_weight("../test_datasets/test_new_lane_train.txt", 4, 8, normalize=False)  # only for test

