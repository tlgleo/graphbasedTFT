import numpy as np

def create_list_names_agents(list_chars):
    ident2char = ['T1', 'T2', 'T3', 'GT1', 'GT2', 'GT3', 'N', 'D', 'Tra', 'Ln']
    char2ident = {x : i for i,x in enumerate(ident2char)}
    ident2name = ['TFT', 'TFT', 'TFT', 'grTFT', 'grTFT', 'grTFT', 'Nice', 'Defector', 'Traitor', 'LateNice']
    ident2cat = [0,0,0,1,1,1,2,3,4,5]
    n_cat = max(ident2cat) + 1
    numero_per_cat = [1] * n_cat

    list_names = []
    for c in list_chars:
        ident = char2ident[c[:3]]
        k_cat = ident2cat[ident]
        name = ident2name[ident] + str(numero_per_cat[k_cat])
        numero_per_cat[k_cat] += 1
        list_names.append(name)

    return list_names


def normalise(a, b, x):
    # transform x from [a,b] to [0,1]
    return np.round((x-a)/(b-a),3)


def transform_list_metrics(list_metrics, list_renorm=[(0,1),(0,1),(0,1),(0,1),(-1,0)]):
    new_metrics = []
    for norm, x in zip(list_renorm, list_metrics):
        (a,b) = norm
        new_metrics.append(normalise(a,b,x))
    return new_metrics

