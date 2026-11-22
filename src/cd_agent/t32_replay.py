"""Build the T3.2 module using the demo_g3 catalog and replay renderers."""
from pathlib import Path
import os

from .make_demo import esc, pct
from .make_g3_pages import audit_html
from .make_site import Store, page, render_replay
from .replay_catalog import render_catalog
from .t32_dev32 import load, save


DESCRIPTION = '单期类别视觉参考（K=1）'


def select_cases(rows):
    positives = sorted((row for row in rows if not row['is_negative']),
                       key=lambda row: (row['f1'], row['iou'], row['case']))
    submitted = [row for row in positives if row['submitted_by'] == 'model']
    upper = submitted[-2:] if len(submitted) >= 2 else positives[-2:]
    used = {row['case'] for row in upper}
    candidates = [row for row in positives if row['case'] not in used]
    middle = min(candidates[2:], key=lambda row: (not 0 < row['recall'] < 1,
                                                  abs(row['f1'] - .5), row['case']))
    worst = [row for row in candidates if row != middle]
    # Prefer distinct failure outcomes where tied on F1.
    first = worst[0]
    second = min(worst[1:], key=lambda row: (row['f1'], (row['fp'] > 0) == (first['fp'] > 0),
                                             row['iou'], row['case']))
    tagged = [('高分', row) for row in reversed(upper)] + [('中等', middle), ('低分', first), ('低分', second)]
    return {row['case']: tag for tag, row in tagged}


def build_replays(out, summary):
    replay = out / 'replay'
    replay.mkdir(exist_ok=True)
    rows = summary['cases']
    chosen = select_cases(rows)
    records = []
    for row in rows:
        case = row['case']
        directory = out / 'traces' / case
        final = load(directory / 'final.json')
        record = dict(page=case, image=case, task='T3.2', dataset=row['dataset'], family='t32',
                      description=DESCRIPTION, version='G3', run='T3.2 Dev-32', source=final['planner_model'],
                      file=f'{case}.html', selection=chosen.get(case, 'Dev-32 会话'), f1=row['f1'],
                      session=str(directory), gt_path='offline-target.png')
        document, _ = render_replay(None, record, Store(replay / '.unused', 'build'))
        audit_html(document)
        (replay / record['file']).write_text(document, encoding='utf-8')
        records.append({key: value for key, value in record.items() if key not in ('session', 'gt_path')})
    ranked = [next(record for record in records if record['image'] == case) for case in chosen]
    summaries = [('全部', summary['overall']), *summary['datasets'].items()]
    stats = ''.join(
        f'<tr><th>{esc(name)}</th><td>{block["count"]}</td><td>{block["negatives"]}</td>'
        + ''.join(f'<td>{pct(block["micro"][key])}</td>' for key in ('iou', 'f1', 'precision', 'recall'))
        + f'<td>{pct(block["positive_macro"]["f1"] if block["positive_macro"] else None)}</td>'
          f'<td>{block["negative_false_alarms"]}</td></tr>' for name, block in summaries)
    case_rows = []
    for record, row in zip(records, rows):
        case_rows.append(f'<tr><td><a href="{record["file"]}">{esc(row["case"])}</a></td>'
                         f'<td>{esc(chosen.get(row["case"], ""))}</td><td>{esc(row["dataset"])}</td>'
                         + ''.join(f'<td>{pct(row[key])}</td>' for key in ('iou', 'f1', 'precision', 'recall'))
                         + f'<td>{row["false_alarm_pixels"] if row["is_negative"] else "—"}</td>'
                           f'<td>{esc(row["submitted_by"])}</td></tr>')
    body = (
        '<header><h1>T3.2 单期类别视觉参考</h1><p>Dev-32 · K=1 · qwen3.7-flash Planner 和 Verifier</p>'
        '<nav><a href="../../cd_agent/demo_g3/index.html">返回多任务回放</a></nav></header>'
        '<main class="hub"><section class="panel"><h2>总体与分来源</h2><div class="tablewrap"><table>'
        '<thead><tr><th>来源</th><th>题数</th><th>空真值</th><th>微平均 IoU</th><th>微平均 F1</th>'
        '<th>微平均 P</th><th>微平均 R</th><th>正例宏平均 F1</th><th>空真值虚警像素</th></tr></thead>'
        f'<tbody>{stats}</tbody></table></div>'
        '<p>真值 1 为前景，0 为背景，255 为忽略像素。预测非零为前景。双空图 F1 和 IoU 为 1，P/R 的零分母记 0。</p>'
        '<p><a href="../metrics.json">下载完整指标</a></p></section>'
        '<section class="panel"><h2>32 条会话</h2><div class="tablewrap"><table><thead><tr>'
        '<th>会话</th><th>代表性</th><th>来源</th><th>IoU</th><th>F1</th><th>Precision</th><th>Recall</th>'
        f'<th>虚警像素</th><th>提交来源</th></tr></thead><tbody>{"".join(case_rows)}</tbody></table></div>'
        '</section></main>')
    document = page('T3.2 Dev-32 回放', body)
    audit_html(document)
    (replay / 'index.html').write_text(document, encoding='utf-8')
    save(out / 'selection.json', ranked)


def catalog_records(out, demo):
    selection = out / 'selection.json'
    if not selection.is_file():
        return [], None
    relative = Path(os.path.relpath(out / 'replay', demo))
    records = [dict(record, page=f't32-{record["page"]}', file=(relative / record['file']).as_posix())
               for record in load(selection)]
    return records, (relative / 'index.html').as_posix()


def merge_into_catalog(out, demo):
    """Update the existing catalog index only, reusing its metadata and renderer."""
    coverage_path = demo / 'coverage.json'
    original = load(coverage_path)
    new, t32_url = catalog_records(out, demo)
    if not new:
        raise ValueError('T3.2 evaluation must finish before publishing its replay cards')
    records = [row for row in original['records'] if row.get('family') != 't32'] + new
    document = render_catalog(records, t32_url)
    report = audit_html(document)
    for href in report['links']:
        target = href.split('#', 1)[0]
        if target and not (demo / target).is_file():
            raise ValueError(f'Broken demo_g3 link: {href}')
    pending = demo / '.t32-index.html'
    pending.write_text(document, encoding='utf-8')
    pending.replace(demo / 'index.html')
    original['records'] = records
    for cell in original['coverage']:
        if cell['task'] == 'T3.2' and cell['constraint'] == 'C0':
            cell.update(status='replay', count=len(new), evidence='T3.2 Dev-32')
    save(coverage_path, original)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--demo', type=Path, required=True)
    args = parser.parse_args()
    merge_into_catalog(args.out.resolve(), args.demo.resolve())
