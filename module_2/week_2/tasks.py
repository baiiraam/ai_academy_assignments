import random
import matplotlib.pyplot as plt

class CustomErrorToShowThatIWantToUseExceptionsError(Exception):
    pass

def plot_graphs(x_data, y_data, x_label, y_label, title, color=None):
    plt.figure()
    # if there are more than one graph to plot in the same figure, same x-axis
    if all(type(arg)==list for arg in [x_data, y_data, y_label]):
        lx = len(x_data)
        if lx > 1 and all(len(arg)==lx for arg in [y_data, y_label]):
            for i in range(lx):
                color=list(random.random() for c in range(3))
                plt.plot(x_data[i], y_data[i], color=color, label=y_label[i])
            plt.xlabel(x_label)
            plt.title(title)
            plt.legend()
        else:
            raise CustomErrorToShowThatIWantToUseExceptionsError("Something went wrong with plotting")
    else:
        plt.plot(x_data, y_data, color=color)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        plt.xlim([min(x_data), max(x_data)])
    plt.grid(True)
    plt.show()


def get_data_csv(fn: str | None=None, colx: int | None=None, coly: int | None=None, sep: str=',') -> tuple:
    '''
    Get csv data.

    Parameters:
    fn: file name STR
    colx: x column INT
    coly: y column INT
    sep: separator STR
    '''
    if fn is None or colx is None or coly is None:
        raise CustomErrorToShowThatIWantToUseExceptionsError("Oops")

    # initiate the return variables, they will be overwritten (probably)
    x_label = ""
    y_label = ""
    x_data = []
    y_data = []
    # will modify later hopefully and probably

    # read file
    with open(fn, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                split_line = line.strip().split(sep)
                x_label = split_line[colx]
                y_label = split_line[coly]
            else:
                split_line = line.strip().split(sep)
                x_data.append(float(split_line[colx]))
                y_data.append(float(split_line[coly]))

    return x_data, y_data, x_label, y_label


if __name__ == "__main__":
    x_label = ""
    user_fn = input("Enter CSV filename(s), separated by ',': ")
    user_colx = input("X cols (separated with ', '): ")
    user_coly = input("Y cols (separated with ', '): ")
    user_title = input("Enter title")
    fn = user_fn.split(', ')
    colx = list(int(c) for c in user_colx.split(', '))
    coly = list(int(c) for c in user_coly.split(', '))
    if len(fn) == 1:
        fn = fn[0]
        x_data, y_data, x_label, y_label = get_data_csv(fn, colx[0], coly[0])
    else:
        x_data, y_data, y_label = [], [], []
        for i in range(len(fn)):
            auxx, auxy, x_label, auxylbl = get_data_csv(fn[i], colx[i], coly[i])
            x_data.append(auxx)
            y_data.append(auxy)
            y_label.append(auxylbl)
    plot_graphs(x_data, y_data, x_label, y_label, user_title, color='black')
