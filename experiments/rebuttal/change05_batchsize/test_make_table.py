"""Fast tests for the table builder."""

from experiments.rebuttal.change05_batchsize import make_table


def model(group, batch_size, auroc, share=1., dead=0):
    entry = dict(latent=0, sel=.9, auroc=auroc, tpr=.5, fpr=.01)
    return dict(group=group, batch_size=batch_size, seed=0,
                features=dict(wheel=dict(auroc=entry, sel=entry)),
                health=dict(dead_latents=dead, variance_explained=.3), match_share=share)


def models(aurocs):
    result = {'paper': model('paper', 1024, .9)}
    for batch_size, values in aurocs.items():
        for seed, value in enumerate(values):
            result[f'{batch_size}_{seed}'] = model('retrained', batch_size, value)
    return result


def test_fmt_mean_and_range():
    assert make_table.fmt([.8, .9, .7]) == '0.800 [0.700, 0.900]'
    assert make_table.fmt([.5, .75], percent=True) == '62 [50, 75]'
    assert make_table.single(.91234) == '0.912'


def test_loss_flags_need_two_of_three_seeds():
    ok = {512: [.9, .88, .91], 1024: [.9, .9, .9], 2048: [.6, .6, .91]}
    assert make_table.loss_flags(models(ok), ['wheel']) == [('wheel', 2048)]
    one = {512: [.5, .9, .9], 1024: [.9, .9, .9], 2048: [.9, .9, .9]}
    assert make_table.loss_flags(models(one), ['wheel']) == []


def test_latex_table_has_all_batch_sizes():
    data = models({512: [.9, .9, .9], 1024: [.9, .9, .9], 2048: [.9, .9, .9]})
    text = make_table.latex([('Aeon', data, ['wheel'])])
    assert text.count('\\begin{tabular*}') == 1 and 'Dead latents' in text and '$B=2{,}048$' in text
    assert text.count('AUROC of best-AUROC latent') == 1
