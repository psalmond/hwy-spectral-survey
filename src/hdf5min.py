# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""
hdf5min.py -- minimal pure-Python HDF5 reader for the MATLAB v7.3 subset:
superblock v0, symbol-table groups (TREE/SNOD/HEAP), v1 object headers,
dataspace v1, datatype classes 0/1 (int/float LE), layout v3 contiguous or
chunked (B-tree v1) with optional deflate filter.  Returns numpy arrays
transposed to MATLAB orientation.  No attributes, no refs, no structs.
"""
import struct, zlib
import numpy as np

U64 = struct.Struct('<Q')
U32 = struct.Struct('<I')
U16 = struct.Struct('<H')

class H5Min:
    def __init__(self, path):
        self.buf = open(path, 'rb').read()
        sig = b'\x89HDF\r\n\x1a\n'
        off = 0
        while not self.buf[off:off+8] == sig:
            off = 512 if off == 0 else off*2
            if off > len(self.buf):
                raise IOError('no HDF5 signature')
        s = off + 8
        ver = self.buf[s]
        if ver != 0:
            raise NotImplementedError(f'superblock v{ver}')
        self.so = self.buf[s+5]   # sizeof offsets
        self.sl = self.buf[s+6]   # sizeof lengths
        assert self.so == 8 and self.sl == 8
        self.base = U64.unpack_from(self.buf, s+16)[0]
        # root group symbol table entry at s+48 (after 4 addrs of 8)
        e = s + 48
        self.root_oh = self.base + U64.unpack_from(self.buf, e+8)[0]

    def addr(self, a):
        return None if a == 0xffffffffffffffff else self.base + a

    # ---------------- object header v1 -> list of messages
    def messages(self, oh):
        b = self.buf
        ver, _, nmsg, _refc, hsize = struct.unpack_from('<BBHII', b, oh)
        assert ver == 1, f'objhdr v{ver}'
        msgs = []
        # first block starts after 16-byte header (12 used + 4 pad)
        blocks = [(oh+16, hsize)]
        while blocks:
            pos, size = blocks.pop(0)
            end = pos + size
            while pos + 8 <= end and len(msgs) < nmsg:
                mtype, msize, _flags = struct.unpack_from('<HHB', b, pos)
                body = pos + 8
                if mtype == 0x0010:   # continuation
                    caddr = self.addr(U64.unpack_from(b, body)[0])
                    clen = U64.unpack_from(b, body+8)[0]
                    blocks.append((caddr, clen))
                else:
                    msgs.append((mtype, body, msize))
                pos = body + msize
        return msgs

    # ---------------- group walk
    def group_links(self, oh):
        out = {}
        li = None
        for mtype, body, _ in self.messages(oh):
            if mtype == 0x0011:           # STAB (old style)
                bt = self.addr(U64.unpack_from(self.buf, body)[0])
                hp = self.addr(U64.unpack_from(self.buf, body+8)[0])
                out.update(self._walk_btree_group(bt, self._heap_data(hp)))
            elif mtype == 0x0006:         # direct Link message
                n, a = self._parse_link(body)
                if n:
                    out[n] = a
            elif mtype == 0x0002:         # Link Info (new style)
                li = body
        if li is not None and not out:
            fh = self.addr(U64.unpack_from(self.buf, li+2)[0])
            if fh is not None:
                out.update(self._fractal_links(fh))
        return out

    def _parse_link(self, p):
        b = self.buf
        if b[p] != 1:
            return None, None
        flags = b[p+1]
        q = p + 2
        ltype = 0
        if flags & 0x08:
            ltype = b[q]; q += 1
        if flags & 0x04:
            q += 8
        if flags & 0x10:
            q += 1
        lsz = 1 << (flags & 0x03)
        nlen = int.from_bytes(b[q:q+lsz], 'little'); q += lsz
        if not (0 < nlen < 256):
            return None, None
        name = b[q:q+nlen]
        q += nlen
        if ltype != 0 or not all(32 <= c < 127 for c in name):
            return None, None
        a = self.addr(U64.unpack_from(b, q)[0])
        if a is None or a + 16 > len(b) or b[a] != 1 or b[a+1] != 0:
            return None, None
        return name.decode(), a

    def _fractal_links(self, fh):
        b = self.buf
        assert b[fh:fh+4] == b'FRHP'
        maxheap = U16.unpack_from(b, fh+128)[0]
        offsz = (maxheap + 7)//8
        flags = b[fh+9]
        rootaddr = self.addr(U64.unpack_from(b, fh+132)[0])
        out = {}
        if rootaddr is None:
            return out
        assert b[rootaddr:rootaddr+4] == b'FHDB', b[rootaddr:rootaddr+4]
        p = rootaddr + 4 + 1 + self.so + offsz
        if flags & 0x02:
            p += 4
        while p < len(b) and b[p] == 1:
            n, a = self._parse_link(p)
            if n is None:
                break
            out[n] = a
            # re-walk to find next record start
            q = p + 2
            fl = b[p+1]
            if fl & 0x08: q += 1
            if fl & 0x04: q += 8
            if fl & 0x10: q += 1
            lsz = 1 << (fl & 0x03)
            nlen = int.from_bytes(b[q:q+lsz], 'little')
            p = q + lsz + nlen + self.so
        return out

    def _heap_data(self, hp):
        assert self.buf[hp:hp+4] == b'HEAP'
        return self.addr(U64.unpack_from(self.buf, hp+24)[0])

    def _walk_btree_group(self, bt, heap_data):
        b = self.buf
        assert b[bt:bt+4] == b'TREE', b[bt:bt+4]
        ntype, level, nent = b[bt+4], b[bt+5], U16.unpack_from(b, bt+6)[0]
        assert ntype == 0
        out = {}
        p = bt + 8 + 2*self.so          # skip left/right siblings
        p += self.sl                     # key 0
        for _ in range(nent):
            child = self.addr(U64.unpack_from(b, p)[0]); p += self.so
            p += self.sl                 # key i+1
            if level > 0:
                out.update(self._walk_btree_group(child, heap_data))
            else:
                out.update(self._snod(child, heap_data))
        return out

    def _snod(self, sn, heap_data):
        b = self.buf
        assert b[sn:sn+4] == b'SNOD'
        nsym = U16.unpack_from(b, sn+6)[0]
        out = {}
        p = sn + 8
        for _ in range(nsym):
            name_off = U64.unpack_from(b, p)[0]
            ohdr = self.addr(U64.unpack_from(b, p+8)[0])
            name_p = heap_data + name_off
            name_e = b.index(b'\x00', name_p)
            out[b[name_p:name_e].decode()] = ohdr
            p += 40
        return out

    # ---------------- dataset read
    def dataset(self, oh):
        dims = dtype = None
        layout = None
        filters = []
        for mtype, body, msize in self.messages(oh):
            b = self.buf
            if mtype == 0x0001:          # dataspace
                ver, rank, flags = b[body], b[body+1], b[body+2]
                p = body + (8 if ver == 1 else 4)
                dims = [U64.unpack_from(b, p+8*i)[0] for i in range(rank)]
            elif mtype == 0x0003:        # datatype
                cv, size = b[body], U32.unpack_from(b, body+4)[0]
                cls = cv & 0x0f
                if cls == 1:
                    dtype = {8: '<f8', 4: '<f4'}[size]
                elif cls == 0:
                    signed = (b[body+1] >> 3) & 1
                    dtype = ('<i' if signed else '<u') + str(size)
                else:
                    raise NotImplementedError(f'dtype class {cls}')
            elif mtype == 0x0008:        # layout v3
                ver, cls = b[body], b[body+1]
                assert ver == 3, f'layout v{ver}'
                if cls == 1:             # contiguous
                    a = self.addr(U64.unpack_from(b, body+2)[0])
                    sz = U64.unpack_from(b, body+10)[0]
                    layout = ('contig', a, sz)
                elif cls == 2:           # chunked
                    nd = b[body+2]
                    bt = self.addr(U64.unpack_from(b, body+3)[0])
                    cd = [U32.unpack_from(b, body+11+4*i)[0]
                          for i in range(nd)]   # last = elem size
                    layout = ('chunk', bt, cd)
                elif cls == 0:           # compact
                    sz = U16.unpack_from(b, body+2)[0]
                    layout = ('compact', body+4, sz)
            elif mtype == 0x000B:        # filter pipeline
                nf = b[body+1]
                p = body + 8
                for _ in range(nf):
                    fid = U16.unpack_from(b, p)[0]
                    nlen = U16.unpack_from(b, p+2)[0]
                    ncv = U16.unpack_from(b, p+6)[0]
                    p += 8 + nlen
                    p += (8 - nlen % 8) % 8 if nlen % 8 else 0
                    p += 4*ncv
                    if ncv % 2:
                        p += 4
                    filters.append(fid)
        if dims is None or layout is None:
            return None
        npdt = np.dtype(dtype)
        n = int(np.prod(dims)) if dims else 1
        if layout[0] == 'contig':
            _, a, sz = layout
            arr = np.frombuffer(self.buf, npdt, n, a)
        elif layout[0] == 'compact':
            _, a, sz = layout
            arr = np.frombuffer(self.buf, npdt, n, a)
        else:
            _, bt, cd = layout
            cdims, esz = cd[:-1], cd[-1]
            assert esz == npdt.itemsize
            full = np.zeros(dims if dims else [1], npdt)
            for off, raw in self._chunks(bt, len(cdims)):
                data = zlib.decompress(raw) if 1 in filters else raw
                c = np.frombuffer(data, npdt, count=int(np.prod(cdims)))
                c = c.reshape(cdims)
                sl = tuple(slice(o, min(o+cs, d))
                           for o, cs, d in zip(off, cdims, dims))
                csl = tuple(slice(0, s.stop - s.start) for s in sl)
                full[sl] = c[csl]
            arr = full
        arr = np.array(arr).reshape(dims if dims else [1])
        return arr.T                     # MATLAB orientation

    def _chunks(self, bt, rank):
        b = self.buf
        assert b[bt:bt+4] == b'TREE', b[bt:bt+4]
        ntype, level, nent = b[bt+4], b[bt+5], U16.unpack_from(b, bt+6)[0]
        assert ntype == 1
        out = []
        p = bt + 8 + 2*self.so
        for _ in range(nent):
            csize = U32.unpack_from(b, p)[0]
            off = [U64.unpack_from(b, p+8+8*i)[0] for i in range(rank)]
            p += 8 + 8*(rank+1)          # key: size,mask,offsets+elem dim
            child = self.addr(U64.unpack_from(b, p)[0]); p += self.so
            if level > 0:
                out.extend(self._chunks(child, rank))
            else:
                out.append((off, b[child:child+csize]))
        return out

def loadmat73(path):
    f = H5Min(path)
    out = {}
    for name, oh in f.group_links(f.root_oh).items():
        if name.startswith('#'):
            continue
        try:
            a = f.dataset(oh)
            if a is not None:
                out[name] = a
        except NotImplementedError as e:
            out[name] = f'<unreadable: {e}>'
    return out

if __name__ == '__main__':
    import sys
    d = loadmat73(sys.argv[1])
    for k, v in d.items():
        print(k, getattr(v, 'shape', v),
              getattr(v, 'dtype', ''),
              (f"[{v.ravel()[0]:.6g} .. {v.ravel()[-1]:.6g}]"
               if hasattr(v, 'ravel') and v.size else ''))
