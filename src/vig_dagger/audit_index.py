"""Navigation and provenance for generated DAgger analysis outputs."""
from collections import Counter
from datetime import datetime, timezone
import json


def write_overview(root, rows):
    dest = root / 'results/dagger_audit'
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines = ['# DAgger analysis — start here', '', f'Inventory updated: {now} (UTC).', '',
             '## Review order', '',
             '1. [Compare selected settings](COMPARISONS.md): compact plots, with separate success/failed/all cohorts.',
             '2. [Collection report](REPORT.md): collection rates and detailed metrics.',
             '3. [Failed attempts](FAILED_ATTEMPTS.md): exact recorded reasons and available videos.',
             '4. [Successful attempts](SUCCESSFUL_ATTEMPTS.md): successful demonstrations and clips.',
             '5. [All videos](VIDEOS.md): live rendering inventory and handoff clips.', '',
             'Collection counts below come from completed episode records. Plot counts reflect the last analysis snapshot and may lag collection.', '',
             '| Setting | Success / attempts | Failed | Full videos ready | Pending | Historical RGB unavailable | Figures updated (UTC) |',
             '|---|---:|---:|---:|---:|---:|---|']
    from vig_dagger.audit import selected_variants
    for variant in sorted(selected_variants(root)):
        rs = [r for r in rows if r['variant'] == variant]
        counts = Counter(r['status'] for r in rs)
        stamp = dest / variant / 'snapshot.json'
        updated = json.loads(stamp.read_text())['generated_at_utc'] if stamp.exists() else 'earlier snapshot'
        success = sum(r['outcome'] == 'SUCCESS' for r in rs)
        if not rs:
            lines.append(f'|{variant}|No completed attempts|—|—|pending|—|—|')
            continue
        lines.append(f"|[{variant}]({variant}/FIGURES.md)|{success}/{len(rs)}|{len(rs)-success}|{counts['ready']}|{counts['pending']}|{sum(v for k,v in counts.items() if k.startswith('unavailable:'))}|{updated}|")
    lines += ['', '## How to interpret the metrics', '',
              '- Pooled expert portion weights each action equally; per-attempt box plots weight each attempt equally.',
              '- An intervention is one continuous run of expert actions; an attempt can contain several interventions.',
              '- Terminal failure/time-limit segments are observed durations, not completed recovery times.',
              '- Stage labels come from simulator state. Final stage, furthest stage and planner failure reason describe different things.',
              '- The expert uses geometric grasp checks and stops on failure without retries. “Dropped” is a detector diagnosis, not proof of an initially secure grasp.',
              '- Only the first target number of successful attempts is selected for training; failed recordings are analysis-only.',
              '- Missing historical RGB cannot be reconstructed. Pending videos have recordings awaiting processing or safe access.', '',
              '## Files', '',
              '`compare_*.png/pdf`: compact comparisons. `<driver>/<variant>/FIGURES.md`: current detailed plots. CSV/JSON files alongside those plots contain the numbers.', '',
              'Older multi-panel PNGs are retained for provenance; use the linked indexes for the current figure set.', '']
    tmp = dest / 'README.md.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.replace(dest / 'README.md')
