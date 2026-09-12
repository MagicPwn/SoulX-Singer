"""Lossless-timeline lyric replacement for SoulX native SVS metadata."""
from __future__ import annotations

import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import math
import unicodedata

_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)*")
MIN_SYLLABLE_SECONDS = Decimal('0.03')


def lyric_tokens(text: str) -> list[list[str]]:
    """Clean/tokenize lines: Han characters or English words, no silent loss.

    Whitespace/punctuation are not sung. Numbers, unknown scripts and symbols
    must be written out in supported words rather than converted to <SP>.
    """
    result = []
    for number, line in enumerate(clean_lyrics(text), 1):
        tokens = _TOKEN.findall(line)
        remaining = _TOKEN.sub('', line)
        if any(not (c.isspace() or unicodedata.category(c).startswith('P')) for c in remaining):
            raise ValueError(f'unsupported sung lyric character on cleaned line {number}; spell out numbers/symbols')
        if not tokens:
            raise ValueError(f'lyric line {number} has no sung content')
        result.append(tokens)
    if not result:
        raise ValueError('lyrics contain no sung content after removing headings')
    return result

_FIELDS = ('text', 'phoneme', 'duration', 'note_pitch', 'note_type')


def _validate(metadata_list):
    if not isinstance(metadata_list, list):
        raise ValueError('metadata must be a list of segment dictionaries')
    parsed = []
    previous_end = 0.0
    for number, segment in enumerate(metadata_list):
        try:
            if not isinstance(segment, dict):
                raise ValueError('must be a dictionary')
            time = segment['time']
            if not isinstance(time, (list, tuple)) or len(time) != 2:
                raise ValueError('time must contain [start_ms, end_ms]')
            start, end = map(float, time)
            if not all(math.isfinite(x) for x in (start, end)) or start < previous_end - 0.001 or end <= start:
                raise ValueError('invalid, overlapping, or out-of-order time range')
            previous_end = end
            if any(not isinstance(segment[k], str) for k in (*_FIELDS, 'f0')):
                raise ValueError('token fields and f0 must be strings')
            fields = {key: segment[key].split() for key in _FIELDS}
            n = len(fields['text'])
            if not n or any(len(v) != n for v in fields.values()):
                raise ValueError('text/phoneme/duration/note_pitch/note_type token counts differ or are empty')
            durations = list(map(Decimal, fields['duration']))
            if any(not d.is_finite() or d <= 0 for d in durations):
                raise ValueError('durations must be finite and positive')
            pitches = list(map(int, fields['note_pitch']))
            types = list(map(int, fields['note_type']))
            if any(not 0 <= p <= 127 for p in pitches):
                raise ValueError('note_pitch must be MIDI 0..127')
            f0 = list(map(float, segment['f0'].split()))
            if not f0 or any(not math.isfinite(x) or x < 0 for x in f0):
                raise ValueError('f0 must contain finite nonnegative values')
            # Native JSON rounds note lengths to two decimals. Never correct or
            # stretch this rounding drift; preserve both time and duration sum.
            tolerance = max(0.03, n * 0.005 + 0.02)
            if abs(float(sum(durations)) - (end - start) / 1000) > tolerance:
                raise ValueError('duration sum disagrees with segment time')
            for i, (word, ph, typ) in enumerate(zip(fields['text'], fields['phoneme'], types)):
                if word == '<SP>':
                    if typ != 1 or ph != '<SP>':
                        raise ValueError('rest must have <SP> phoneme and note_type 1')
                elif ph == '<SP>' or typ not in (2, 3):
                    raise ValueError('sung token must have a phoneme and note_type 2/3')
                elif typ == 3 and (i == 0 or types[i - 1] == 1 or word != fields['text'][i - 1] or ph != fields['phoneme'][i - 1]):
                    raise ValueError('orphan/inconsistent note_type 3 tie')
            fields['durations'] = durations
            fields['types'] = types
            # Explicit provenance warning, NOT a measured ASR confidence score.
            warning = str(segment.get('warning', '')).lower()
            fields['needs_review'] = bool(segment.get('review') or segment.get('scaffold_notes')
                                          or 'asr' in warning or 'scaffold' in warning)
            parsed.append(fields)
        except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError) as exc:
            raise ValueError(f'metadata segment {number}: {exc}') from exc
    return parsed


def _map_run(fields, indices, tokens, phonemes):
    """Intersect target syllable spans with all original note boundaries.

    Unflagged source onsets and their type-3 notes form expressive slots.
    Explicitly warned/scaffolded runs use cumulative actual sung time instead:
    inferred word counts cannot squeeze syllables into tiny source slots.
    Every original note is retained (possibly subdivided) in either mode.
    """
    boundaries = [Decimal(0)]
    groups = []
    for i in indices:
        if fields['types'][i] == 2 or not groups:
            groups.append(boundaries[-1])
        boundaries.append(boundaries[-1] + fields['durations'][i])
    groups.append(boundaries[-1])
    cuts = [Decimal(0)]
    slots, count = len(groups) - 1, len(tokens)
    for k in range(1, count):
        if fields['needs_review']:
            cut = boundaries[-1] * k / count
        else:
            coordinate = Decimal(k * slots) / count
            slot = int(coordinate)
            cut = groups[slot] + (coordinate - slot) * (groups[slot + 1] - groups[slot])
        cuts.append(cut.quantize(Decimal('0.000000001')))
    cuts.append(boundaries[-1])
    if any(b - a < MIN_SYLLABLE_SECONDS for a, b in zip(cuts, cuts[1:])):
        raise ValueError('unachievable alignment: a sung lyric syllable would be shorter than 30 ms')
    edits = {}
    k = 0
    previous = None
    for j, index in enumerate(indices):
        left, right = boundaries[j:j + 2]
        pieces = []
        while left < right:
            while cuts[k + 1] <= left:
                k += 1
            end = min(right, cuts[k + 1])
            duration = fields['duration'][index] if left == boundaries[j] and end == right else format(end - left, 'f')
            pieces.append((tokens[k], phonemes[k], duration, fields['note_pitch'][index], '3' if k == previous else '2'))
            previous = k
            left = end
        edits[index] = pieces
    return edits


def _source_phrases(parsed, target_count):
    phrases = []
    for si, fields in enumerate(parsed):
        phrase = None
        i = 0
        rest = Decimal(0)
        while i < len(fields['text']):
            if fields['text'][i] == '<SP>':
                rest += fields['durations'][i]
                i += 1
                continue
            if phrase is None or rest >= Decimal('0.25'):
                phrase = {'segment': si, 'runs': [], 'slots': 0, 'notes': 0, 'capacity': 0,
                          'active_seconds': 0.0, 'needs_review': fields['needs_review']}
                phrases.append(phrase)
            rest = Decimal(0)
            run = []
            while i < len(fields['text']) and fields['text'][i] != '<SP>':
                run.append(i)
                i += 1
            slots = sum(fields['types'][j] == 2 for j in run)
            seconds = sum(fields['durations'][j] for j in run)
            capacity = int(seconds / MIN_SYLLABLE_SECONDS)
            phrase['runs'].append({'indices': run, 'slots': slots, 'capacity': capacity, 'segment': si,
                                   'active_seconds': float(seconds), 'needs_review': fields['needs_review']})
            phrase['slots'] += slots
            phrase['notes'] += len(run)
            phrase['capacity'] += capacity
            phrase['active_seconds'] += float(seconds)
    # Calibrate scaffold weights in source-slot units from unflagged material,
    # or from target density when no source onsets are trustworthy. Silence is
    # excluded, and source onset counts never weight a flagged run/phrase.
    reliable = [p for p in phrases if not p['needs_review']]
    seconds = sum(p['active_seconds'] for p in reliable or phrases)
    rate = (sum(p['slots'] for p in reliable) if reliable else target_count) / seconds if seconds else 0
    for phrase in phrases:
        for run in phrase['runs']:
            run['effective_slots'] = run['active_seconds'] * rate if run['needs_review'] else run['slots']
        phrase['effective_slots'] = sum(run['effective_slots'] for run in phrase['runs'])
        phrase['effective_notes'] = phrase['effective_slots'] if phrase['needs_review'] else phrase['notes']
    return phrases


def _line_groups(phrases, lines):
    """Monotone DP: 1:N lines or N:1 phrases, without omissions/reordering."""
    pcount, lcount = len(phrases), len(lines)
    prefix = {key: [0] for key in ('effective_slots', 'effective_notes', 'capacity', 'minimum')}
    for phrase in phrases:
        for key in prefix:
            value = len(phrase['runs']) if key == 'minimum' else phrase[key]
            prefix[key].append(prefix[key][-1] + value)
    target = [0]
    for line in lines:
        target.append(target[-1] + len(line))
    # Normalize the mixed reliable/scaffold weights to the requested lyric
    # count. Preserve the existing count-based DP for wholly unflagged sources.
    scale = target[-1] / prefix['effective_slots'][-1] if any(p['needs_review'] for p in phrases) else 1.0
    best = [[math.inf] * (lcount + 1) for _ in range(pcount + 1)]
    back = {}
    best[0][0] = 0.0
    for p in range(pcount):
        for l in range(lcount):
            if not math.isfinite(best[p][l]):
                continue
            choices = [(p + 1, end) for end in range(l + 1, lcount + 1)]
            choices.extend((end, l + 1) for end in range(p + 2, pcount + 1))
            for pe, le in choices:
                n = target[le] - target[l]
                if not prefix['minimum'][pe] - prefix['minimum'][p] <= n <= prefix['capacity'][pe] - prefix['capacity'][p]:
                    continue
                slots = (prefix['effective_slots'][pe] - prefix['effective_slots'][p]) * scale
                notes = (prefix['effective_notes'][pe] - prefix['effective_notes'][p]) * scale
                cost = math.log((n + 0.5) / (slots + 0.5)) ** 2 * max(pe - p, le - l)
                cost += 0.025 * math.log((n + 0.5) / (notes + 0.5)) ** 2
                cost += 0.18 * (pe - p + le - l - 2)
                candidate = best[p][l] + cost
                if candidate < best[pe][le]:
                    best[pe][le] = candidate
                    back[pe, le] = (p, l, cost)
    if not math.isfinite(best[pcount][lcount]):
        raise ValueError('unachievable alignment: lyrics cannot cover every voiced span within duration limits; review line breaks/ASR')
    groups = []
    p, l = pcount, lcount
    while p or l:
        pp, ll, cost = back[p, l]
        groups.append((pp, p, ll, l, cost))
        p, l = pp, ll
    return list(reversed(groups))


def _distribute(runs, count):
    """Integer DP allocation to non-rest runs, never tie across a breath."""
    scale = count / sum(run['effective_slots'] for run in runs)
    best = {0: (0.0, [])}
    for run in runs:
        next_best = {}
        ideal = run['effective_slots'] * scale
        for used, (cost, counts) in best.items():
            for n in range(1, min(run['capacity'], count - used) + 1):
                candidate = cost + ((n - ideal) / max(1, ideal)) ** 2
                if used + n not in next_best or candidate < next_best[used + n][0]:
                    next_best[used + n] = (candidate, counts + [n])
        best = next_best
    if count not in best:
        raise ValueError('unachievable alignment: fewer lyric syllables than disjoint voiced spans, or excessive syllable rate')
    return best[count][1]


def assign_lyrics(metadata_list, lyrics_text, language='Mandarin') -> list[dict]:
    """Deep-copy and replace lyrics, keeping every segment's complete timeline.

    None/'' requests source-lyric conversion. An explicitly supplied blank,
    heading-only or unsupported text raises ValueError. Source phrases split
    at >=250 ms of consecutive <SP> and at segment boundaries. Whole lyric
    lines align with a monotone DP; line/phrase groupings are allowed.
    Explicit review/scaffold warnings switch the affected phrase/run weights
    and syllable cuts to active sung duration (never silence). Unflagged runs
    retain source onset/tie timing. This is a fallback heuristic, not evidence
    of transcript accuracy or a scientifically calibrated confidence score.
    """
    parsed = _validate(metadata_list)
    result = deepcopy(metadata_list)
    if lyrics_text is None or lyrics_text == '':
        return result
    lines = lyric_tokens(lyrics_text)
    phrases = _source_phrases(parsed, sum(map(len, lines)))
    if not phrases:
        raise ValueError('no sung slots in metadata; transcription may have failed (this does not imply no vocals)')
    groups = _line_groups(phrases, lines)
    from preprocess.tools.g2p import g2p_transform
    edits = [{} for _ in result]
    assigned_lines = [set() for _ in result]
    for ps, pe, ls, le, cost in groups:
        tokens = [token for line in lines[ls:le] for token in line]
        line_numbers = [li + 1 for li in range(ls, le) for _ in lines[li]]
        phonemes = g2p_transform(tokens, language)
        runs = [run for phrase in phrases[ps:pe] for run in phrase['runs']]
        counts = _distribute(runs, len(tokens))
        offset = 0
        for run, count in zip(runs, counts):
            si = run['segment']
            edits[si].update(_map_run(parsed[si], run['indices'], tokens[offset:offset + count], phonemes[offset:offset + count]))
            assigned_lines[si].update(line_numbers[offset:offset + count])
            offset += count
    for si, (segment, fields) in enumerate(zip(result, parsed)):
        rows = []
        for i in range(len(fields['text'])):
            rows.extend(edits[si].get(i, [tuple(fields[key][i] for key in _FIELDS)]))
        for column, key in enumerate(_FIELDS):
            segment[key] = ' '.join(row[column] for row in rows)
        seconds = float(sum(d for d, t in zip(fields['durations'], fields['types']) if t != 1))
        syllables = sum(row[4] == '2' for row in rows)
        segment['lyric_alignment'] = {
            # One-based cleaned lyric line numbers; a line spanning multiple
            # segments appears in each segment that actually received tokens.
            'assigned_lines': sorted(assigned_lines[si]),
            'assigned_syllables': syllables,
            'active_seconds': seconds,
            'syllables_per_second': syllables / seconds if seconds else 0.0,
            'needs_review': fields['needs_review'],
            'timing_basis': ('active_sung_time' if fields['needs_review'] else 'source_slots') if seconds else 'silence',
            'warning': ('Source review/scaffold warning: active-sung-time heuristic; '
                        'check lyric timing manually. Not a transcript confidence score.') if fields['needs_review'] else '',
        }
    return result
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET


def read_lyrics(path) -> str:
    """Read UTF-8 TXT or DOCX locally; no office application or uploads."""
    path = Path(path)
    if path.suffix.lower() == '.txt':
        return path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.docx':
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read('word/document.xml'))
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        paragraphs = []
        for paragraph in root.iter(ns + 'p'):
            parts = []
            for node in paragraph.iter():
                if node.tag == ns + 't':
                    parts.append(node.text or '')
                elif node.tag in (ns + 'br', ns + 'cr'):
                    parts.append('\n')
                elif node.tag == ns + 'tab':
                    parts.append('\t')
            paragraphs.append(''.join(parts))
        return '\n'.join(paragraphs)
    if path.suffix.lower() == '.doc':
        return _read_doc(path)
    raise ValueError('Unsupported lyric file: use UTF-8 .txt, .docx, or .doc')


def _read_doc(path: Path) -> str:
    """Read Word 97–2003 main story using MS-DOC FIB/CLX piece offsets.

    Unlike scanning printable bytes, the piece table preserves editing order
    and excludes stale/deleted buffers, headers, and document properties.
    """
    import struct
    try:
        import olefile
    except ImportError as exc:
        raise RuntimeError('Legacy .doc requires olefile: install requirements-longform.txt') from exc
    try:
        with olefile.OleFileIO(str(path)) as ole:
            word = ole.openstream('WordDocument').read()
            if len(word) < 154 or struct.unpack_from('<H', word)[0] != 0xA5EC:
                raise ValueError('Invalid Word document FIB')
            version, = struct.unpack_from('<H', word, 2)
            flags, = struct.unpack_from('<H', word, 10)
            if version < 0xC1:
                raise ValueError('Word before 97 is unsupported; save as DOCX or UTF-8 TXT')
            if flags & (0x100 | 0x8000):
                raise ValueError('Encrypted Word documents are unsupported; supply an unencrypted copy')
            table = ole.openstream('1Table' if flags & 0x200 else '0Table').read()
        csw, = struct.unpack_from('<H', word, 32)
        lw_pos = 34 + csw * 2
        cslw, = struct.unpack_from('<H', word, lw_pos)
        if cslw < 4:
            raise ValueError('Invalid Word FIB long-word section')
        text_length, = struct.unpack_from('<I', word, lw_pos + 2 + 3 * 4)
        fc_pos = lw_pos + 2 + cslw * 4
        count, = struct.unpack_from('<H', word, fc_pos)
        if count < 34:
            raise ValueError('Word document has no CLX piece table')
        start, size = struct.unpack_from('<II', word, fc_pos + 2 + 33 * 8)
        if size == 0 or start + size > len(table):
            raise ValueError('Invalid or missing Word CLX piece table')
        clx = table[start:start + size]
        pos = 0
        while pos < len(clx) and clx[pos] == 1:
            length, = struct.unpack_from('<H', clx, pos + 1)
            pos += 3 + length
        if pos >= len(clx) or clx[pos] != 2:
            raise ValueError('Invalid Word piece table marker')
        length, = struct.unpack_from('<I', clx, pos + 1)
        plc = clx[pos + 5:pos + 5 + length]
        if len(plc) != length or length < 4 or (length - 4) % 12:
            raise ValueError('Truncated Word piece table')
        n = (length - 4) // 12
        cps = struct.unpack_from(f'<{n + 1}I', plc)
        if cps[0] != 0 or cps[-1] < text_length or any(a > b for a, b in zip(cps, cps[1:])):
            raise ValueError('Invalid Word story coverage')
        parts = []
        for i in range(n):
            if cps[i] >= text_length:
                break
            chars = min(cps[i + 1], text_length) - cps[i]
            fc, = struct.unpack_from('<I', plc, (n + 1) * 4 + i * 8 + 2)
            compressed = bool(fc & 0x40000000)
            offset = fc & 0x3FFFFFFF
            if compressed:
                offset //= 2
            byte_count = chars * (1 if compressed else 2)
            raw = word[offset:offset + byte_count]
            if len(raw) != byte_count:
                raise ValueError('Truncated Word text piece')
            parts.append(raw.decode('cp1252' if compressed else 'utf-16le'))
        text = ''.join(parts)
        # Keep paragraph, soft-line, page and table-cell boundaries.
        return text.replace('\r', '\n').replace('\x0b', '\n').replace('\x0c', '\n').replace('\x07', '\n')
    except (OSError, KeyError, struct.error, UnicodeError, IndexError) as exc:
        raise ValueError('Cannot extract legacy DOC; it is malformed or uses an unsupported encoding') from exc

_SECTION = re.compile(
    r'(?:verse|chorus|pre[- ]?chorus|bridge|intro|outro|interlude|instrumental|'
    r'refrain|hook|主歌|副歌|预副歌|前副歌|桥段|过渡|前奏|间奏|尾奏|尾声|合唱)'
    r'(?:\s*[0-9一二三四五六七八九十ABⅠⅡⅢ]+)?', re.IGNORECASE)


def clean_lyrics(text: str) -> list[str]:
    """Remove standalone titles/section labels, never substring-match lyrics.

    Punctuation in sung lines is retained here; the aligner tokenizes it later.
    """
    if not isinstance(text, str):
        raise TypeError('lyrics text must be a string')
    lines = []
    for raw in text.replace('\ufeff', '').splitlines():
        line = raw.strip()
        if not line or re.fullmatch(r'《[^《》\n]+》', line):
            continue
        label = line.strip('[]【】()（）:： ').strip()
        if _SECTION.fullmatch(label):
            continue
        lines.append(line)
    return lines
