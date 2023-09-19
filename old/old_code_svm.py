def get_metrics(dataset):
    not_null = [ p for p in dataset if p[0][0] is not None and p[0][1] is not None ]

    return {
        'not_null': not_null,
        'not_null_hit': [ p for p in not_null if p[1] == 1 ],
        'not_null_nohit': [ p for p in not_null if p[1] == 0 ]
    }

def show_filled_persons(queues, indices):
    queues_no = len(queues)

    for i in range(queues_no):
        for j in range(queues_no):
            persons = load_persons(queues, indices, i, j)
            print(len([ p for p in persons if p[0][0] is not None and p[0][1] is not None ]), end='')
            print(' ', end='')
        print()

def get_persons_by_indices(frames, indices):
    return [ flatten_fighters_list(frames[i]) for i in indices ]