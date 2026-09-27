"""Per-file PyAAF2 compatibility; upstream classes are never monkeypatched."""
# Portions adapted from PyAAF2, Copyright (c) 2017 Mark Reid (MIT).
# See the adjacent PYAAF2_LICENSE.txt notice.
from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from aaf2.cfb import CompoundFileBinary, Stream
from aaf2.file import AAFFile, AAFObjectManager, AAFFactory
from aaf2.metadict import MetaDictionary


class _LenientStream(Stream):
    def write(self, data):
        data_size = len(data)
        current_size = self.dir.byte_size
        new_size = max(self.tell() + data_size, current_size)
        if new_size > current_size:
            self.allocate(new_size)

        mv = memoryview(data)

        is_mini_stream = self.dir.byte_size < self.storage.min_stream_max_size
        full_sector_size = self.storage.sector_size
        mini_sector_size = self.storage.mini_stream_sector_size
        sector_cache = self.storage.sector_cache
        f = self.storage.f

        if is_mini_stream:
            mini_fat_index = self.pos // mini_sector_size
            mini_sector_offset = self.pos % mini_sector_size
            sector_size = mini_sector_size
        else:
            index = self.pos // full_sector_size
            sid_offset = self.pos % full_sector_size
            sector_size = full_sector_size

        while data_size > 0:
            if is_mini_stream:
                mini_stream_sid = self.fat_chain[mini_fat_index]
                mini_stream_pos = (mini_stream_sid * mini_sector_size) + mini_sector_offset
                index = mini_stream_pos // full_sector_size
                sid_offset = mini_stream_pos % full_sector_size
                while index >= len(self.storage.mini_stream_chain):
                    self.storage.mini_stream_grow()
                sid = self.storage.mini_stream_chain[index]
                sector_offset = mini_sector_offset
                seek_pos = ((sid + 1) * full_sector_size) + sid_offset
                mini_fat_index += 1
                mini_sector_offset = 0
            else:
                sid = self.fat_chain[index]
                sector_offset = sid_offset
                seek_pos = ((sid + 1) * full_sector_size) + sid_offset
                index += 1
                sid_offset = 0

            byte_writeable = min(len(mv), sector_size - sector_offset)
            if byte_writeable <= 0:
                break
            if sid in sector_cache:
                del sector_cache[sid]
            f.seek(seek_pos)
            f.write(mv[:byte_writeable])
            self.pos += byte_writeable
            mv = mv[byte_writeable:]
            data_size -= byte_writeable
        return None


class _LenientStorage(CompoundFileBinary):
    def read_dir_entry(self, dir_id, parent=None):
        try:
            return super().read_dir_entry(dir_id, parent)
        except IndexError:
            return None

    def open(self, path, mode='r'):
        stream = super().open(path, mode)
        return _LenientStream(self, stream.dir, mode)


class _InertReadMetaDictionary(MetaDictionary):
    def read_properties(self):
        previous_mode = self.root.mode
        if previous_mode == 'rb+':
            self.root.mode = 'rb'
        try:
            return super().read_properties()
        finally:
            self.root.mode = previous_mode


class _LenientAAFFile(AAFFile):
    """PyAAF2 initialization with explicit storage/metadata factories.

    PyAAF2 has no factory injection in AAFFile.__init__. Keep this small adapter
    aligned with that constructor; all object reading and saving remain upstream.
    """
    def __init__(self, path, mode, sector_size):
        modes = {'r': 'rb', 'rb': 'rb', 'r+': 'rb+', 'rb+': 'rb+',
                 'rw': 'rb+', 'w': 'wb+', 'w+': 'wb+', 'wb+': 'wb+'}
        if mode not in modes:
            raise ValueError(f'Invalid AAF mode: {mode}')
        self.mode = modes[mode]
        self.is_open = False
        self.f = io.open(path, self.mode)
        try:
            self.cfb = _LenientStorage(self.f, self.mode, sector_size=sector_size)
            self.weakref_table = []
            self.manager = AAFObjectManager(self)
            self.create = AAFFactory(self)
            if self.mode in ('rb', 'rb+'):
                self.read_reference_properties()
                self.metadict = _InertReadMetaDictionary(self)
                self.metadict.dir = self.cfb.find('/MetaDictionary-1')
                self.manager['/MetaDictionary-1'] = self.metadict
                self.root = self.manager.read_object('/')
                self.metadict.read_properties()
            else:
                self.setup_empty()
            self.is_open = True
        except BaseException:
            self.f.close()
            raise


@contextmanager
def open_aaf_lenient(path: Path | str, mode: str) -> Iterator[AAFFile]:
    """Open with instance-scoped CFB compatibility and observable save failures."""
    try:
        aaf = _LenientAAFFile(str(path), mode, sector_size=4096)
    except (IndexError, ValueError):
        # Only malformed-container errors justify a second opening strategy.
        aaf = _LenientAAFFile(str(path), mode, sector_size=512)
    try:
        yield aaf
    finally:
        try:
            aaf.close()
        finally:
            # A failed save must still release the OS handle without retrying it.
            aaf.is_open = False
            aaf.f.close()
