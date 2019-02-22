import os
from data_IO.data_reader import read_csv


def traverse_folder(folders):
    event_dict = {}
    for folder in folders:
        for f in os.listdir(folder):
            event_count = 0
            event_id = []
            f_path = os.path.join(folder, f)

            # parse data
            data = read_csv(f_path)

            # data dict
            FORWARD_VELOCITY = 8
            FORWARD_ACCELERATION = 4
            THRESH = -0.4
            # detect event
            last_i = 0
            for i in range(1, data.shape[0]):
                d = data[i, FORWARD_ACCELERATION]
                if d != ' ' and float(d) < THRESH and i - last_i > 40:
                    last_i = i
                    event_count += 1
                    event_id.append(i)
            event_dict[f_path] = event_id
            print('Event count in %s is %i' % (f, event_count))
    return event_dict


def process_event_dict(ed):
    """

    :param ed: ed is short for event dictionary.
            Key is file path
            Value is a list of event time
    :return:
    """
    if len(ed) == 0:
        return None
    for (k, v) in ed.items():
        event_name = k.split('\\')[-1]
        with open(os.path.join('./vtti_event_id', event_name + '.txt'), 'w') as out:
            for i in v:
                out.write(str(i) + '\n')


if __name__ == '__main__':
    folders = [r'E:\journal_paper\Applied_Science\data\VehicleID_296344_DriverID_22207\csv']
    data_dict = traverse_folder(folders)
    process_event_dict(data_dict)
