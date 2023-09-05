import model.enn_model
import time
import utils.log_tool
import utils.dataset_tools
import utils.read_cfg_tool

if __name__ == "__main__":
    utils.log_tool.log_function_settlement()  # recording terminal output
    output_num, feature_num, learning_epochs, learning_rate = \
        utils.read_cfg_tool.read_yaml_data(cfg_path="./cfg/datasets.yaml")  # get training configs from .yaml

    train_dataset_path, weight_path = utils.dataset_tools.find_init_weight("dataset/fire_train.txt"
                                                                           , output_num, feature_num, normalize=False)  # init enn weight values

    my_enn = model.enn_model.my_enn(train_dataset_path,
                                    weight_path,
                                    feature_num,
                                    output_num)

    time_start = time.process_time()  # time start

    my_enn.enn_forward(epoch=learning_epochs,
                       learning_rate=learning_rate)  # start training model

    time_end = time.process_time()  # time over
    time_sum = time_end - time_start  # counting time by using /s
    print("total training time: {} s".format(time_sum))

    test_path = "dataset/fire_test.txt"
    my_enn.enn_test(test_dataset_path=test_path)  # test function
