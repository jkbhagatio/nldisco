"""Paper-only typography helpers; never change numerical result values."""

import re

METRICS = ('AUROC', 'Sel', 'TPR', 'FPR', 'MSE', 'MSLE', 'L0')
FEATURES = (
    'recent deceleration', 'fast leftward target reach',
    'early leftward movement', 'late, target-specific reach',
    'wheel-active association', 'smaller segmented area near the patch',
    'highest centroid-speed quintile', 'radially outward heading during fast movement',
    'smaller segmented area near patch', 'smaller area near patch',
    'highest speed quintile', 'fast radially outward', 'wheel association',
    'wheel active', 'wheel-associated', 'wheel-active',
    'radially outward movement', 'radially outward',
    'deceleration', 'early-leftward', 'late-reaching',
    'fast-reach', 'fast-target', 'late-target',
)
_FEATURE_RE = re.compile(
    r'(?<![\w\\])(?:' + '|'.join(re.escape(f) for f in FEATURES) + r')(?!\w)',
    re.IGNORECASE,
)
_METRIC_RE = re.compile(
    r'(?<!\w)(' + '|'.join(METRICS) + r')(s?)(?!\w)'
)


def format_tex(text: str, features: bool = True) -> str:
    """Style printed labels while leaving references, paths and existing italics alone."""
    for metric in METRICS:
        for command in ('mathrm', 'mathit', 'mathbf', 'textit', 'textbf'):
            text = text.replace('\\' + command + '{' + metric + '}',
                                '\\metric{' + metric + '}')
    text = text.replace(r'R\textsuperscript{2}', r'\metric{R^2}')
    # Definition-list headings: italicize the name, not the configuration or punctuation.
    if features:
        for feature in sorted(FEATURES, key=len, reverse=True):
            text = re.sub(r'\\textbf\{(' + re.escape(feature) + r')(:?)\}',
                          lambda m: r'\textit{' + m[1] + '}' + m[2], text,
                          flags=re.IGNORECASE)
            text = re.sub(r'\\textbf\{(' + re.escape(feature) + r') (\([^{}]*\)\.)\}',
                          lambda m: r'\textit{' + m[1] + '} ' + m[2], text,
                          flags=re.IGNORECASE)
    protected = []

    def hold(match):
        protected.append(match[0])
        return f'@@PROTECTED{len(protected)-1}@@'

    text = re.sub(
        r'\\(?:metric|textit|emph|path|texttt|label|input|includegraphics|autoref|ref)'
        r'(?:\[[^\]]*\])?\{[^{}]*\}|https?://[^\s}]+|(?m:^%[^\n]*)', hold, text)
    text = _METRIC_RE.sub(lambda m: r'\metric{' + m[1] + '}' + m[2], text)
    if features:
        text = _FEATURE_RE.sub(lambda m: r'\textit{' + m[0] + '}', text)
    for i, value in enumerate(protected):
        text = text.replace(f'@@PROTECTED{i}@@', value)
    return text


def format_plot_label(label: str) -> str:
    """Use Matplotlib mathtext for bold metric symbols and italic feature names."""
    # Protect existing math segments, including labels styled on a prior save.
    pieces = re.split(r'(\$[^$]*\$)', label)
    for i in range(0, len(pieces), 2):
        value = _METRIC_RE.sub(lambda m: r'$\mathbf{' + m[1] + '}$' + m[2], pieces[i])
        value = _FEATURE_RE.sub(
            lambda m: r'$\mathit{' + m[0].replace(' ', r'\ ') + '}$', value)
        pieces[i] = value
    return ''.join(pieces)
