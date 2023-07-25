import torch
import matplotlib.pyplot as plt
import os

class Logger(object):
    def __init__(self, runs, info=None, patience=100):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.best_valid_acc = 0.0
        self.count = 0
        self.patience = patience
    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)
        return False
        # if result[1] > self.best_valid_acc:
        #     self.best_valid_acc = result[1]
        #     self.count = 0
        # else:
        #     self.count += 1
        # if self.count >= self.patience:
        #     self.count = 0
        #     self.best_valid_acc = 0.0
        #     return True
        # else:
        #     return False

    def print_statistics(self, run=None, base_dir_name='train_results', model_name='model_name', sub_dir_name_prefix='sub_dir_name'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            # result = [[(i[0]*100, i[1]*100, i[2]*100) for i in r] for r in self.results]


            best_results = []
            base_dir_name = os.path.join(base_dir_name, 'train_results')
            if not os.path.exists(base_dir_name):
                os.mkdir(base_dir_name)
            dir_name = model_name
            dir_name = os.path.join(base_dir_name, dir_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            sub_dir_index = 1
            sub_dir_name = sub_dir_name_prefix+str(sub_dir_index)
            while os.path.exists(os.path.join(dir_name, sub_dir_name)):
                sub_dir_index += 1
                sub_dir_name = sub_dir_name_prefix+str(sub_dir_index)
            os.mkdir(os.path.join(dir_name, sub_dir_name))
            result_dir = os.path.join(dir_name, sub_dir_name)
            runtime = 0
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                # train_acc = [i[0] for i in r]
                # valid_acc = [i[1] for i in r]
                # test_acc = [i[2] for i in r]

                # train1 = max(train_acc)
                # valid = max(valid_acc)
                # valid_max_index = valid_acc.index(valid)
                # train2 = r[valid_max_index][0]
                # test = r[valid_max_index][2]

                best_results.append((train1, valid, train2, test))
                epochs = list(range(1, len(r) + 1))
                train_acc = r[:, 0]
                valid_acc = r[:, 1]
                test_acc = r[:, 2]
                plt.figure()
                plt.plot(epochs, train_acc, label='Train Accuracy')
                plt.plot(epochs, valid_acc, label='Valid Accuracy')
                plt.plot(epochs, test_acc, label='Test Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.title(f'{model_name} Accuracy')
                plt.legend()
                runtime += 1
                plt.savefig(os.path.join(result_dir, f'{model_name}_run{runtime}_accuracy.png'))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')