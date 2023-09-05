import yaml


def read_yaml_data(cfg_path: str) -> (int, int, int, float):
    with open(cfg_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    output_num = yaml_data['datasetData']['output_num']
    feature_number = yaml_data['datasetData']['feature_number']
    training_epochs = yaml_data['learning_set']['epochs']
    learning_rate = yaml_data['learning_set']['learning_rate']
    return output_num, feature_number, training_epochs, learning_rate
