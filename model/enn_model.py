import numpy as np
import os
import time
import prettytable as pt


class my_enn:
    """
    使用ENN进行数据的学习和推理
    """
    def __init__(self,
                 train_dataset_path: str,
                 weight_path: str,
                 feature_num: int,
                 output_num: int) -> None:
        # init x, y axis data
        self.x = []
        self.y = []

        self.datasetPath = train_dataset_path
        self.weightPath = weight_path
        self.feature_num = feature_num
        self.output_num = output_num

        self.testDataMat = []
        self.testLabelMat = []
        self.dataMat = []
        self.labelMat = []
        self.floatLine = []

        self.weight_init_mat = np.ones((1, 1, 1))
        self.zCenter = np.ones((1, 1))

    # read in training data
    def train_dataset_load(self) -> None:
        file = open(self.datasetPath)
        for line in file.readlines():
            curLine = line.strip().split(" ")
            floatLine = list(map(float, curLine))  # using map func to trans data into type float
            self.dataMat.append(floatLine[0:self.feature_num])
            self.labelMat.append(floatLine[-1])
        print("Training dataset read in successfully!")

    # read in test data
    def test_dataset_load(self, test_path) -> None:
        file = open(test_path)
        for line in file.readlines():
            curLine = line.strip().split(" ")
            floatLine = list(map(float, curLine))  # using map func to trans data into type float
            self.testDataMat.append(floatLine[0:self.feature_num])
            self.testLabelMat.append(floatLine[-1])
        print("Testing dataset read in successfully!")
        print('\n')

    # initialize weight from .txt file
    def init_weight(self) -> None:
        featureNum = (len(self.dataMat[0]))
        print(self.dataMat[0])

        print("featureNum is: {}".format(featureNum))
        print("outputNum is: {}".format(self.output_num))
        self.weight_init_mat = np.ones((self.output_num, featureNum, 2))
        self.zCenter = np.ones((self.output_num, featureNum))
        print("weight_init_mat shape is: {}".format(self.weight_init_mat.shape))
        file = open(self.weightPath)
        for line in file.readlines():
            curLine = line.strip().split(" ")
            floatLine = list(map(float, curLine))  # using map func to trans data into type float
            self.floatLine.append(floatLine)
        print(len(self.floatLine))

        for i in range(0, self.output_num, 1):
            for j in range(0, featureNum, 1):
                print("i, j : {}, {}".format(i, j))
                self.weight_init_mat[i][j] = self.floatLine[i][j * 2:j * 2 + 2]
                print('self.weight_init_mat[{}][{}] : {}'.format(i, j, self.weight_init_mat[i][j]))
                self.zCenter[i][j] = (self.weight_init_mat[i][j][0] + self.weight_init_mat[i][j][1]) / 2
                print('self.zCenter[{}][{}] : {}'.format(i, j, self.zCenter[i][j]))
        print('weight_init_mat is: {}'.format(self.weight_init_mat))

    def ED_counting(self, x: float, z: float, Wu: float, Wl: float) -> float:
        wu_wl_2 = (Wu - Wl) / 2
        ans = (abs(x - z) - wu_wl_2) / abs(wu_wl_2)
        ans = ans + 1
        return ans

    # if pred != label, renew weight values
    def weight_renew(self, learning_rate: float, zWrong: float, zRight: float, wUWrong: float,
                     wLWrong: float, wURight: float, wLRight: float, x: float) \
                    ->(float, float, float, float, float, float):

        zRightNew = zRight + learning_rate * (x - zRight)
        zWrongNew = zWrong - learning_rate * (x - zWrong)

        wLRightNew = wLRight + learning_rate * (x - zRight)
        wURightNew = wURight + learning_rate * (x - zRight)

        wLWrongNew = wLWrong + learning_rate * (x - zWrong)
        wUWrongNew = wUWrong + learning_rate * (x - zWrong)

        return zWrongNew, zRightNew, wUWrongNew, wLWrongNew, wURightNew, wLRightNew

    def enn_forward(self, epoch: int, learning_rate: float) -> None:
        self.train_dataset_load()
        self.init_weight()
        featureNum = (len(self.dataMat[0]))
        outputNum = self.output_num
        Em = len(self.dataMat)
        correction_times = 0

        # start forward counting
        for i in range(epoch):
            Er = 0
            if i % 10 == 0:
                print('epoch : {} / {} started!'.format(i, epoch))

            for index, trainData in enumerate(self.dataMat):
                ED_list = []
                for j in range(outputNum):
                    # counting every kind of outputs
                    features_ans = 0
                    for q in range(featureNum):  # counting ED between object's features and classes
                        # counting every features by using ED function
                        ans = self.ED_counting(trainData[q], self.zCenter[j][q], self.weight_init_mat[j][q][0],
                                               self.weight_init_mat[j][q][1])
                        features_ans += ans

                    ED_list.append(features_ans)

                class_pred = int(np.argmin(ED_list))
                class_label = int(self.labelMat[index])

                if class_pred != int(self.labelMat[index]):
                    Er += 1
                    correction_times += 1
                    for w in range(featureNum):
                        self.zCenter[class_pred][w], self.zCenter[class_label][w], \
                        self.weight_init_mat[class_pred][:, 0][w], self.weight_init_mat[class_pred][:, 1][w], \
                        self.weight_init_mat[class_label][:, 0][w], self.weight_init_mat[class_label][:, 1][w] = \
                            self.weight_renew(
                                              learning_rate,
                                              self.zCenter[class_pred][w], self.zCenter[class_label][w],
                                              self.weight_init_mat[class_pred][:, 0][w],
                                              self.weight_init_mat[class_pred][:, 1][w],
                                              self.weight_init_mat[class_label][:, 0][w],
                                              self.weight_init_mat[class_label][:, 1][w],
                                              trainData[w])
            print("-"*50)
            print('epoch {} error rate is: {}'.format(i, (Er / Em)))
            print('Er: {}, Em: {}'.format(Er, Em))
            if i % 1 == 0:
                self.x.append(i)
                self.y.append(Er / Em)
            if (Er / Em) < 0.0001:
                break

        print('correction_times is: {}'.format(correction_times))
        print('new Weight is: {}'.format(self.weight_init_mat))
        print('new zCenter is: {}'.format(self.zCenter))
        print('\n')
        self.show_train_cfg()

    def show_train_cfg(self) -> None:
        import matplotlib.pyplot as plt
        plt.axis([0, 100, 0, 1.0])  # （0, 100）range of x， （0, 1.0）range of y
        plt.xticks([i * 10 for i in range(0, 11)], fontsize=26)  # show x-axis
        plt.yticks([i * 0.1 for i in range(0, 11)], fontsize=26)  # show y-axis
        plt.plot(self.x, self.y, color="r", linestyle="-", linewidth=4, label="label")
        plt.title("Learning error convergence curves", color='k', fontsize=26)
        plt.ylabel("Learning error rate", fontsize=26)  # name of y-axis
        plt.xlabel("Learning epoch", fontsize=26)  # name of x-axis
        plt.show()

    def enn_test(self, test_dataset_path) -> None:
        if not os.path.exists(test_dataset_path):
            print("--can not find test file--")
            return
        self.test_dataset_load(test_dataset_path)

        Er = 0
        Em = len(self.testLabelMat)

        tb = pt.PrettyTable()
        tb.field_names = ["Test data idx", "Predict type", "True type", "Prediction status"]
        time_start = time.process_time()  # record time start

        for index, fireData in enumerate(self.testDataMat):
            ED_list = []
            featureNum = self.feature_num
            outputNum = self.output_num

            for j in range(outputNum):
                # counting every kind of outputs
                features_ans = 0
                for q in range(featureNum):
                    # counting every feature come through ED function
                    ans = self.ED_counting(fireData[q], self.zCenter[j][q], self.weight_init_mat[j][q][0],
                                           self.weight_init_mat[j][q][1])
                    features_ans += ans
                ED_list.append(features_ans)

            class_pred = int(np.argmin(ED_list))
            detect_s = "[Prediction correct]"

            if class_pred != int(self.testLabelMat[index]):
                detect_s = "[**Prediction error**]"
                Er += 1
            tb.add_row([index + 1, (class_pred + 1), int(self.testLabelMat[index]), detect_s])

        time_end = time.process_time()  # record time end
        time_sum = time_end - time_start  # counting time coast
        print(tb)
        print('\n')

        # show test results
        print('=' * 30, 'Test result', '=' * 30)
        print('Number of test samples: {}'.format(len(self.testDataMat)))
        print('Number of correct predictions: {}'.format(Em - Er))
        correct_times = round(((Em - Er) / Em), 2) * 100
        print('Correct prediction rate: {}%'.format(correct_times))
        print("total test time: {} s, hz: {} s".format(time_sum, time_sum / float(Em)))
