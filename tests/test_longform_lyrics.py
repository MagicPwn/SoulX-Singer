"""Synthetic tests only: no user song/lyric content is stored here."""
import importlib
import unittest
import tempfile
import zipfile
import struct
from pathlib import Path
from copy import deepcopy


def metadata(words=('原', '曲'), durations=('0.4', '0.6'), pitches=('60', '62'), types=('2', '2'), start=0):
    from decimal import Decimal
    return {
        'time': [start, start + float(sum(map(Decimal, durations))) * 1000],
        'text': ' '.join(words),
        'phoneme': ' '.join('<SP>' if x == '<SP>' else 'zh_yuan2' for x in words),
        'duration': ' '.join(durations), 'note_pitch': ' '.join(pitches), 'note_type': ' '.join(types),
        'f0': '0.0 261.63 277.1800 0.0', 'extra': {'unchanged': [1, 2]},
    }


class AlignmentTests(unittest.TestCase):
    def test_equal_length_uses_native_g2p_without_changing_timeline(self):
        source = [metadata(('<SP>', '原', '曲', '<SP>'), ('0.2', '0.4', '0.6', '0.5'), ('0', '60', '62', '0'), ('1', '2', '2', '1'))]
        before = deepcopy(source)
        result = lyrics.assign_lyrics(source, '春风')
        self.assertEqual(result[0]['text'], '<SP> 春 风 <SP>')
        self.assertEqual(result[0]['phoneme'], '<SP> zh_chun1 zh_feng1 <SP>')
        for key in ('time', 'f0', 'duration', 'note_pitch', 'note_type'):
            self.assertEqual(result[0][key], source[0][key])
        self.assertEqual(source, before)

    def test_changed_character_count_keeps_every_pitch_interval(self):
        from decimal import Decimal
        for target in ('春', '春风明月', '春春春'):
            source = [metadata()]
            result = lyrics.assign_lyrics(source, target)[0]
            self.assertEqual(''.join(w for w, t in zip(result['text'].split(), result['note_type'].split()) if t == '2'), target)
            self.assertEqual(result['time'], source[0]['time'])
            self.assertEqual(result['f0'], source[0]['f0'])
            by_pitch = {}
            for pitch, dur in zip(result['note_pitch'].split(), result['duration'].split()):
                by_pitch[pitch] = by_pitch.get(pitch, Decimal(0)) + Decimal(dur)
            self.assertEqual(by_pitch, {'60': Decimal('.4'), '62': Decimal('.6')})
            self.assertEqual(len(result['text'].split()), len(result['phoneme'].split()))

    def test_rejects_empty_invalid_and_unachievable_sung_content(self):
        for text in ('  \n', '《测试》\n[Chorus]', '！？——', '春123', '风🙂', '<SP>', '春' * 100):
            with self.subTest(text_length=len(text)):
                with self.assertRaisesRegex(ValueError, 'lyric|alignment|sung|unsupported'):
                    lyrics.assign_lyrics([metadata()], text)
        rest = metadata(('<SP>',), ('1.0',), ('60',), ('1',))
        with self.assertRaisesRegex(ValueError, 'no sung slots.*transcription'):
            lyrics.assign_lyrics([rest], '春风')
        with self.assertRaisesRegex(ValueError, 'no sung slots'):
            lyrics.assign_lyrics([], '春风')

    def test_line_alignment_uses_phrase_slots_not_even_time(self):
        source = [metadata(('原', '曲', '<SP>', '原', '来', '明', '天', '<SP>', '向', '前'),
                           ('0.8', '0.8', '0.3', '0.1', '0.1', '0.1', '0.1', '0.3', '0.2', '0.2'),
                           ('60', '62', '0', '64', '65', '67', '69', '0', '71', '72'),
                           ('2', '2', '1', '2', '2', '2', '2', '1', '2', '2'))]
        output = lyrics.assign_lyrics(source, '春风来\n明月照\n向前')[0]
        phrases = ['']
        for word, typ in zip(output['text'].split(), output['note_type'].split()):
            if word == '<SP>':
                phrases.append('')
            elif typ == '2':
                phrases[-1] += word
        self.assertEqual(phrases, ['春风来', '明月照', '向前'])

    def test_dp_permits_both_line_and_phrase_groupings(self):
        source2 = [metadata(), metadata(('原', '来', '明', '天'), ('0.2',) * 4, ('60', '62', '64', '65'), ('2',) * 4, start=1500)]
        for text in ('春\n风\n明月照我', '春风明月照我'):
            grouped = lyrics.assign_lyrics(source2, text)
            self.assertEqual(grouped[0]['text'], '春 风')
            self.assertEqual(grouped[1]['text'], '明 月 照 我')
            self.assertEqual([x['time'] for x in grouped], [x['time'] for x in source2])

    def test_review_scaffolds_allocate_whole_lines_by_active_time(self):
        # Same sung time, wildly different detected onset counts. The middle
        # segment has a long silence which must not increase its lyric share.
        source = []
        start = 0
        for count, duration, rest in ((20, '0.4', '0.5'), (2, '4', '40'), (40, '0.2', '0.5')):
            item = metadata(('<SP>',) + ('原',) * count + ('<SP>',),
                            (rest,) + (duration,) * count + ('0.5',),
                            ('0',) + ('60',) * count + ('0',),
                            ('1',) + ('2',) * count + ('1',), start=start)
            item['review'] = 'ASR漏词，按检测音符对齐歌词，请逐段检查'
            source.append(item)
            start = item['time'][1]
        before = deepcopy(source)
        lines = ['春风明月照我心', '天高云淡山川向前行', '清风吹过田野',
                 '万家灯火照亮长长归途', '春风明月山川大地', '天高云淡一路向前']
        output = lyrics.assign_lyrics(source, '\n'.join(lines))
        sung = [''.join(w for w, t in zip(s['text'].split(), s['note_type'].split()) if t == '2') for s in output]
        self.assertEqual(sung, [''.join(lines[i:i + 2]) for i in range(0, 6, 2)])
        self.assertEqual(''.join(sung), ''.join(lines))
        self.assertEqual(source, before)
        for original, mapped in zip(source, output):
            self.assertEqual(mapped['time'], original['time'])
            self.assertEqual(mapped['f0'], original['f0'])
            self.assertEqual(mapped['review'], original['review'])
            self.assertEqual([d for d, t in zip(mapped['duration'].split(), mapped['note_type'].split()) if t == '1'],
                             [d for d, t in zip(original['duration'].split(), original['note_type'].split()) if t == '1'])

    def test_warned_runs_use_sung_time_without_short_syllables(self):
        from decimal import Decimal
        # The first run has a tiny onset plus a long missed-ASR note. A short
        # breath joins it to a second run with ten times the onset count.
        item = metadata(('原', '曲', '<SP>') + ('原',) * 20,
                        ('0.04', '3.96', '0.2') + ('0.2',) * 20,
                        ('60', '62', '0') + ('64',) * 20,
                        ('2', '2', '1') + ('2',) * 20)
        target = '春风明月山川大地天高云淡一路向前'
        for flag in ({'review': 'ASR漏词，请检查'}, {'warning': 'ASR scaffold; review'}, {'scaffold_notes': 20}):
            with self.subTest(flag=flag):
                source = dict(item, **flag)
                output = lyrics.assign_lyrics([source], target)[0]
                syllables, runs = [], ['']
                mapped_intervals = []
                clock = Decimal(0)
                for word, typ, dur, pitch in zip(*(output[k].split() for k in ('text', 'note_type', 'duration', 'note_pitch'))):
                    duration = Decimal(dur)
                    mapped_intervals.append((clock, clock + duration, pitch, word == '<SP>'))
                    clock += duration
                    if typ == '1':
                        runs.append('')
                    elif typ == '2':
                        runs[-1] += word
                        syllables.append(duration)
                    else:
                        syllables[-1] += duration
                self.assertEqual(runs, [target[:8], target[8:]])
                for duration in syllables:
                    self.assertAlmostEqual(float(duration), 0.5, places=8)
                # Every original interval survives at exactly its location and
                # pitch, even if the syllable intersects/subdivides that note.
                clock = Decimal(0)
                for word, dur, pitch in zip(*(source[k].split() for k in ('text', 'duration', 'note_pitch'))):
                    end = clock + Decimal(dur)
                    pieces = [x for x in mapped_intervals if clock <= x[0] < end]
                    self.assertEqual(pieces[0][0], clock)
                    self.assertEqual(pieces[-1][1], end)
                    self.assertTrue(all(x[2:] == (pitch, word == '<SP>') for x in pieces))
                    clock = end
                self.assertEqual(output['time'], source['time'])
                self.assertEqual(output['f0'], source['f0'])
                lyrics.assign_lyrics([output], None)  # validates tie/phoneme consistency

    def test_alignment_diagnostics_explain_mixed_source_timing(self):
        reliable = metadata(('原', '原', '曲'), ('0.15', '0.65', '0.2'),
                            ('60', '62', '64'), ('2', '3', '2'))
        flagged = metadata(start=1000)
        flagged['review'] = 'ASR scaffold; check manually'
        silent = metadata(('<SP>',), ('2',), ('0',), ('1',), start=2000)
        source = [reliable, flagged, silent]
        before = deepcopy(source)
        for target, expected_lines in (('春风\n明月', [[1], [2], []]), ('春风明月', [[1], [1], []])):
            with self.subTest(target=target):
                output = lyrics.assign_lyrics(source, target)
                # Reliable onset and its expressive tie keep their exact times,
                # even when another segment needs the duration fallback.
                for key in ('duration', 'note_pitch', 'note_type', 'time', 'f0'):
                    self.assertEqual(output[0][key], reliable[key])
                self.assertEqual(output[0]['text'], '春 春 风')
                for i, (seconds, count, review) in enumerate(((1, 2, False), (1, 2, True), (0, 0, False))):
                    diagnostic = output[i]['lyric_alignment']
                    self.assertEqual(diagnostic['assigned_lines'], expected_lines[i])
                    self.assertEqual(diagnostic['active_seconds'], seconds)
                    self.assertEqual(diagnostic['assigned_syllables'], count)
                    self.assertEqual(diagnostic['syllables_per_second'], count / seconds if seconds else 0)
                    self.assertIs(diagnostic['needs_review'], review)
                self.assertEqual(output[0]['lyric_alignment']['timing_basis'], 'source_slots')
                self.assertEqual(output[1]['lyric_alignment']['timing_basis'], 'active_sung_time')
                self.assertIn('heuristic', output[1]['lyric_alignment']['warning'])
        self.assertEqual(source, before)

    def test_invalid_metadata_is_explicit(self):
        invalid = [
            ('text', '原'), ('duration', '-0.4 1.4'), ('duration', 'NaN 0.6'),
            ('note_pitch', '60 128'), ('note_type', '3 2'), ('time', [1000, 0]),
            ('f0', '0 NaN'), ('phoneme', '<SP> zh_qu3'), ('time', [0, 10000]),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                item = metadata()
                item[key] = value
                with self.assertRaisesRegex(ValueError, 'metadata|segment'):
                    lyrics.assign_lyrics([item], '春风')
        for bad in ({}, [None], [metadata(), metadata()]):
            with self.assertRaisesRegex(ValueError, 'metadata|segment'):
                lyrics.assign_lyrics(bad, '春风')

    def test_no_lyrics_is_conversion_deep_copy(self):
        self.assertTrue(hasattr(lyrics, 'assign_lyrics'), 'assigner is not implemented')
        source = [metadata()]
        for text in (None, ''):
            result = lyrics.assign_lyrics(source, text)
            self.assertEqual(result, source)
            self.assertIsNot(result[0]['extra'], source[0]['extra'])


def synthetic_doc(path):
    """Minimal Word97 OLE with two noncontiguous mixed-encoding pieces."""
    word = bytearray(4096)
    struct.pack_into('<HH', word, 0, 0xA5EC, 0xC1)
    struct.pack_into('<H', word, 10, 0x200)  # 1Table
    struct.pack_into('<H', word, 32, 14)
    struct.pack_into('<H', word, 62, 22)
    struct.pack_into('<I', word, 76, 7)  # main story characters
    struct.pack_into('<H', word, 152, 93)
    word[2048:2058] = '春风\r明月'.encode('utf-16le')
    word[3072:3074] = b'!\r'
    plc = struct.pack('<III', 0, 5, 7)
    plc += struct.pack('<HIH', 0, 2048, 0)
    plc += struct.pack('<HIH', 0, 0x40000000 | (3072 * 2), 0)
    clx = b'\x01\x02\x00xx\x02' + struct.pack('<I', len(plc)) + plc
    struct.pack_into('<II', word, 154 + 33 * 8, 512, len(clx))
    table = bytearray(4096)
    table[512:512 + len(clx)] = clx
    free, end = 0xFFFFFFFF, 0xFFFFFFFE
    header = bytearray(512)
    header[:8] = bytes.fromhex('d0cf11e0a1b11ae1')
    struct.pack_into('<HHHH', header, 24, 0x3E, 3, 0xFFFE, 9)
    struct.pack_into('<H', header, 32, 6)
    struct.pack_into('<IIIIIIIII', header, 40, 0, 1, 0, 0, 4096, end, 0, end, 0)
    struct.pack_into('<109I', header, 76, 17, *([free] * 108))
    directory = bytearray(512)
    for slot, name, kind, sibling, child, start, size in [
        (0, 'Root Entry', 5, free, 1, end, 0),
        (1, 'WordDocument', 2, 2, free, 1, 4096),
        (2, '1Table', 2, free, free, 9, 4096),
    ]:
        off = slot * 128
        encoded = (name + '\0').encode('utf-16le')
        directory[off:off + len(encoded)] = encoded
        struct.pack_into('<HBBIII', directory, off + 64, len(encoded), kind, 1, free, sibling, child)
        struct.pack_into('<IQ', directory, off + 116, start, size)
    fat = [end] + list(range(2, 9)) + [end] + list(range(10, 17)) + [end, 0xFFFFFFFD]
    fat += [free] * (128 - len(fat))
    path.write_bytes(header + directory + word + table + struct.pack('<128I', *fat))

from longform import lyrics


class ReadLyricsTests(unittest.TestCase):
    def test_reads_legacy_doc_piece_table_in_story_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.doc'
            synthetic_doc(path)
            self.assertEqual(lyrics.read_lyrics(path), '春风\n明月!\n')

    def test_reads_utf8_bom_and_docx_paragraphs_and_breaks(self):
        self.assertTrue(hasattr(lyrics, 'read_lyrics'), 'reader is not implemented')
        with tempfile.TemporaryDirectory() as folder:
            txt = Path(folder) / 'lyrics.txt'
            txt.write_bytes('测试\n春风'.encode('utf-8-sig'))
            self.assertEqual(lyrics.read_lyrics(txt), '测试\n春风')
            docx = Path(folder) / 'lyrics.docx'
            with zipfile.ZipFile(docx, 'w') as archive:
                archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>春</w:t></w:r><w:r><w:t>风</w:t><w:br/><w:t>明月</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>向前</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')
            self.assertEqual(lyrics.read_lyrics(docx), '春风\n明月\n向前')


class CleanLyricsTests(unittest.TestCase):
    def test_removes_only_standalone_headings(self):
        self.assertIsNotNone(importlib.util.find_spec('longform.lyrics'), 'lyrics module is not implemented')
        from longform.lyrics import clean_lyrics
        self.assertEqual(clean_lyrics('《测试之歌》\n\n[Verse 1]\n春风吹来\n副歌：\n我们向前\n副歌也在心中\n【Bridge】\n啊——啊！'),
                         ['春风吹来', '我们向前', '副歌也在心中', '啊——啊！'])


if __name__ == '__main__':
    unittest.main()
