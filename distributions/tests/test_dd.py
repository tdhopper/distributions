from nose.tools import assert_equal
from distributions.models import dd_py, dd_cy, dd_cc


VERSIONS = [dd_py, dd_cy, dd_cc]
MODELS = [v.Model for v in VERSIONS]


def test_methods_run():
    alphas = [0.2, 0.5, 1.0, 2.0]
    dim = len(alphas)
    data = [3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1]

    raw_model = {'alphas': alphas}
    models = []
    for Model in MODELS:
        model = Model()
        model.load(raw_model)
        Model.load_model(raw_model)
        models.append(model)

    raw_group = {'counts': [0] * dim}
    for model in models:
        group = model.Group()
        group.load(raw_group)
        group2 = model.load_group(raw_group)

        model.group_init(group)
        for value in data:
            model.group_add_data(group, value)
        model.sample_value(group)
        model.score_group(group)
        for value in data:
            model.score_value(group, value)
        model.group_merge(group2, group)
        assert_equal(group2.dump(), group.dump())
        for value in data:
            model.group_remove_data(group, value)

        model.dump_group(group)
        model.dump()
