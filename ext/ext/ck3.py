#!/usr/bin/env python3

'''
python3 this_script.py -c y3/config.yaml

or

python3 this_script.py y3/config.yaml
'''

import os
import re
import sys
import yaml


########################################################################
#                                                                      #
# resource                                                             #
#                                                                      #
########################################################################


rc_alphabets = {
    '0' : 'a', '1' : 'b', '2' : 'c', '3' : 'd', '4' : 'e',
    '5' : 'f', '6' : 'g', '7' : 'h', '8' : 'i', '9' : 'j',
    '10': 'k', '11': 'l', '12': 'm', '13': 'n', '14': 'o',
    '15': 'p', '16': 'q', '17': 'r', '18': 's', '19': 't',
    '20': 'u', '21': 'v', '22': 'w', '23': 'x', '24': 'y',
    '25': 'z'}

rc_kernel11 = '''
TEMPLATE_PARAMS __global__
STATIC_KERNEL void KERNEL_NAME(SINGLE_LOOP_LIMIT_PARAM EXTRA_KERNEL_PARAMS)
{
    KERNEL_SINGLE_LOOP_BEGIN
    KERNEL_SINGLE_LOOP_CODE
    KERNEL_SINGLE_LOOP_END
}
'''

rc_kernel23c = '''
TEMPLATE_PARAMS            \
__global__                 \
void KERNEL_NAMEc(         \
    TINKER_IMAGE_PARAMS    \
    COUNT_KERNEL_PARAMS    \
    ENERGY_KERNEL_PARAMS   \
    VIRIAL_KERNEL_PARAMS   \
    GRADIENT_KERNEL_PARAMS \
    CUT_KERNEL_PARAMS      \
    OFF_KERNEL_PARAMS      \
    EXTRA_KERNEL_PARAMS    \
    EXCLUDE_SCALE_KERNEL_PARAMS)
{
    USING_DEVICE_VARIABLES    KERNEL_CONSTEXPR_FLAGS \
    const int ithread = threadIdx.x + blockIdx.x * blockDim.x;

    DECLARE_ZERO_LOCAL_COUNT    DECLARE_ZERO_LOCAL_ENERGY    DECLARE_ZERO_LOCAL_VIRIAL
    DECLARE_FORCE_I_AND_K       DECLARE_PARAMS_I_AND_K

    for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
        KERNEL_SCALED_KLANE    KERNEL_ZERO_LOCAL_FORCE

        int i = exclude[ii][0];
        int k = exclude[ii][1];
        KERNEL_LOAD_1X_SCALES

        KERNEL_INIT_EXCLUDE_PARAMS_I_AND_K

        constexpr bool incl = true;
        KERNEL_SCALED_PAIRWISE_INTERACTION

        KERNEL_SAVE_LOCAL_FORCE
    }

    KERNEL_SUM_COUNT    KERNEL_SUM_ENERGY    KERNEL_SUM_VIRIAL
}
'''


rc_kernel23b = '''
TEMPLATE_PARAMS                \
__global__                     \
void KERNEL_NAMEb(             \
    TINKER_IMAGE_PARAMS        \
    COUNT_KERNEL_PARAMS        \
    ENERGY_KERNEL_PARAMS       \
    VIRIAL_KERNEL_PARAMS       \
    GRADIENT_KERNEL_PARAMS     \
    CUT_KERNEL_PARAMS          \
    OFF_KERNEL_PARAMS          \
    EXTRA_KERNEL_PARAMS        \
    EXCLUDE_INFO_KERNEL_PARAMS \
    , const Spatial::SortedAtom* restrict sorted, int n, int nakpl, const int* restrict iakpl LOCAL_KERNEL_PARAMS)
{
    USING_DEVICE_VARIABLES    KERNEL_CONSTEXPR_FLAGS \
    const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
    const int iwarp = ithread / WARP_SIZE;
    const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
    const int ilane = threadIdx.x & (WARP_SIZE - 1);

    DECLARE_ZERO_LOCAL_COUNT    DECLARE_ZERO_LOCAL_ENERGY   DECLARE_ZERO_LOCAL_VIRIAL
    DECLARE_PARAMS_I_AND_K      DECLARE_FORCE_I_AND_K

    for (int iw = iwarp; iw < nakpl; iw += nwarp) {
        KERNEL_ZERO_LOCAL_FORCE

        int tri, tx, ty;
        tri = iakpl[iw];
        tri_to_xy(tri, tx, ty);

        int iid = ty * WARP_SIZE + ilane;
        int atomi = min(iid, n - 1);
        int i = sorted[atomi].unsorted;
        int kid = tx * WARP_SIZE + ilane;
        int atomk = min(kid, n - 1);
        int k = sorted[atomk].unsorted;
        KERNEL_INIT_PARAMS_I_AND_K
        KERNEL_SYNCWARP

        KERNEL_LOAD_INFO_VARIABLES
        for (int j = 0; j < WARP_SIZE; ++j) {
            int srclane = (ilane + j) & (WARP_SIZE - 1); \
            KERNEL_KLANE1                                \
            bool incl = iid < kid and kid < n;           \
            KERNEL_EXCLUDE_BIT                           \
            KERNEL_SCALE_1                               \
            KERNEL_FULL_PAIRWISE_INTERACTION

            iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
            KERNEL_SHUFFLE_PARAMS_I    KERNEL_SHUFFLE_LOCAL_FORCE_I
        }

        KERNEL_SAVE_LOCAL_FORCE   KERNEL_SYNCWARP
    }

    KERNEL_SUM_COUNT    KERNEL_SUM_ENERGY    KERNEL_SUM_VIRIAL
}
'''


rc_kernel23a = '''
TEMPLATE_PARAMS            \
__global__                 \
void KERNEL_NAMEa(         \
    TINKER_IMAGE_PARAMS    \
    COUNT_KERNEL_PARAMS    \
    ENERGY_KERNEL_PARAMS   \
    VIRIAL_KERNEL_PARAMS   \
    GRADIENT_KERNEL_PARAMS \
    CUT_KERNEL_PARAMS      \
    OFF_KERNEL_PARAMS      \
    EXTRA_KERNEL_PARAMS    \
    , const Spatial::SortedAtom* restrict sorted, int niak, const int* restrict iak, const int* restrict lst LOCAL_KERNEL_PARAMS)
{
    USING_DEVICE_VARIABLES    KERNEL_CONSTEXPR_FLAGS
    const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
    const int iwarp = ithread / WARP_SIZE;
    const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
    const int ilane = threadIdx.x & (WARP_SIZE - 1);

    DECLARE_ZERO_LOCAL_COUNT    DECLARE_ZERO_LOCAL_ENERGY   DECLARE_ZERO_LOCAL_VIRIAL
    DECLARE_PARAMS_I_AND_K      DECLARE_FORCE_I_AND_K

    for (int iw = iwarp; iw < niak; iw += nwarp) {
        KERNEL_ZERO_LOCAL_FORCE

        int ty = iak[iw];
        int atomi = ty * WARP_SIZE + ilane;
        int i = sorted[atomi].unsorted;
        int atomk = lst[iw * WARP_SIZE + ilane];
        int k = sorted[atomk].unsorted;
        KERNEL_INIT_PARAMS_I_AND_K
        KERNEL_SYNCWARP

        for (int j = 0; j < WARP_SIZE; ++j) {
            KERNEL_KLANE2          \
            bool incl = atomk > 0; \
            KERNEL_SCALE_1         \
            KERNEL_FULL_PAIRWISE_INTERACTION

            KERNEL_SHUFFLE_PARAMS_I    KERNEL_SHUFFLE_LOCAL_FORCE_I
        }

        KERNEL_SAVE_LOCAL_FORCE    KERNEL_SYNCWARP
    }

    KERNEL_SUM_COUNT    KERNEL_SUM_ENERGY    KERNEL_SUM_VIRIAL
}
'''


rc_kernel21 = '''
TEMPLATE_PARAMS                 \
__global__                      \
void KERNEL_NAME(int n,         \
    TINKER_IMAGE_PARAMS         \
    COUNT_KERNEL_PARAMS         \
    ENERGY_KERNEL_PARAMS        \
    VIRIAL_KERNEL_PARAMS        \
    GRADIENT_KERNEL_PARAMS      \
    CUT_KERNEL_PARAMS           \
    OFF_KERNEL_PARAMS           \
    EXCLUDE_INFO_KERNEL_PARAMS  \
    EXCLUDE_SCALE_KERNEL_PARAMS \
    , const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl
    , int niak, const int* restrict iak, const int* restrict lst
    SORTED_KERNEL_PARAMS LOCAL_KERNEL_PARAMS EXTRA_KERNEL_PARAMS)
{
    USING_DEVICE_VARIABLES    KERNEL_CONSTEXPR_FLAGS \
    const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
    const int iwarp = ithread / WARP_SIZE;
    const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
    const int ilane = threadIdx.x & (WARP_SIZE - 1);

    DECLARE_ZERO_LOCAL_COUNT    DECLARE_ZERO_LOCAL_ENERGY   DECLARE_ZERO_LOCAL_VIRIAL
    DECLARE_PARAMS_I_AND_K      DECLARE_FORCE_I_AND_K

    KERNEL_HAS_1X_SCALE
    for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
        KERNEL_SCALED_KLANE    KERNEL_ZERO_LOCAL_FORCE

        int i = exclude[ii][0];
        int k = exclude[ii][1];
        KERNEL_LOAD_1X_SCALES

        KERNEL_INIT_EXCLUDE_PARAMS_I_AND_K

        constexpr bool incl = true;
        KERNEL_SCALED_PAIRWISE_INTERACTION

        KERNEL_SAVE_LOCAL_FORCE
    }
    // */

    for (int iw = iwarp; iw < nakpl; iw += nwarp) {
        KERNEL_ZERO_LOCAL_FORCE

        int tri, tx, ty;
        tri = iakpl[iw];
        tri_to_xy(tri, tx, ty);

        int iid = ty * WARP_SIZE + ilane;
        int atomi = min(iid, n - 1);
        int i = sorted[atomi].unsorted;
        int kid = tx * WARP_SIZE + ilane;
        int atomk = min(kid, n - 1);
        int k = sorted[atomk].unsorted;
        KERNEL_INIT_PARAMS_I_AND_K
        KERNEL_SYNCWARP

        KERNEL_LOAD_INFO_VARIABLES
        for (int j = 0; j < WARP_SIZE; ++j) {
            int srclane = (ilane + j) & (WARP_SIZE - 1); \
            KERNEL_KLANE1                                \
            bool incl = iid < kid and kid < n;           \
            KERNEL_EXCLUDE_BIT                           \
            KERNEL_SCALE_1                               \
            KERNEL_FULL_PAIRWISE_INTERACTION

            iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
            KERNEL_SHUFFLE_PARAMS_I    KERNEL_SHUFFLE_LOCAL_FORCE_I
        }

        KERNEL_SAVE_LOCAL_FORCE   KERNEL_SYNCWARP
    }

    for (int iw = iwarp; iw < niak; iw += nwarp) {
        KERNEL_ZERO_LOCAL_FORCE

        int ty = iak[iw];
        int atomi = ty * WARP_SIZE + ilane;
        int i = sorted[atomi].unsorted;
        int atomk = lst[iw * WARP_SIZE + ilane];
        int k = sorted[atomk].unsorted;
        KERNEL_INIT_PARAMS_I_AND_K
        KERNEL_SYNCWARP

        for (int j = 0; j < WARP_SIZE; ++j) {
            KERNEL_KLANE2          \
            bool incl = atomk > 0; \
            KERNEL_SCALE_1         \
            KERNEL_FULL_PAIRWISE_INTERACTION

            KERNEL_SHUFFLE_PARAMS_I    KERNEL_SHUFFLE_LOCAL_FORCE_I
        }

        KERNEL_SAVE_LOCAL_FORCE    KERNEL_SYNCWARP
    }

    KERNEL_SUM_COUNT
    KERNEL_SUM_ENERGY
    KERNEL_SUM_VIRIAL
}
'''


# LOCAL_TILES: a tile whose atoms lie within half a box of its center images
# them once around the center, keeping them relative to it, and takes plain
# distances, its virial built from the forces summed over the tile; the other
# tiles image every pair.
# LOCAL_VIRIAL: symmetric takes the symmetric part of that virial, for pair
# forces off the line of centers whose pair virials are symmetrized. The virial
# takes the one scale of the forces, and its lambda derivative the dlsink scale.
def _local_tiles(kernel:str, nloop:int) -> str:
    heads = ['        KERNEL_LOAD_INFO_VARIABLES\n        for (int j = 0; j < WARP_SIZE; ++j) {',
             '        for (int j = 0; j < WARP_SIZE; ++j) {\n            KERNEL_KLANE2']
    found = [head for head in heads if head in kernel]
    if len(found) != nloop:
        raise ValueError('LOCAL_TILES found {} of the {} tile loops of a kernel template.'.format(len(found), nloop))
    for head in found:
        i0 = kernel.index(head)
        i1 = kernel.index('        KERNEL_SAVE_LOCAL_FORCE', i0)
        loop = kernel[i0:i1]
        pre = ''
        if loop.startswith('        KERNEL_LOAD_INFO_VARIABLES\n'):
            pre = '        KERNEL_LOAD_INFO_VARIABLES\n'
            loop = loop[len(pre):]
        local = loop.replace('KERNEL_FULL_PAIRWISE_INTERACTION', 'KERNEL_LOCAL_PAIRWISE_INTERACTION')
        block = pre + '        KERNEL_LOCAL_FLAG\n        if (ilocal) {\n            KERNEL_LOCAL_IMAGE\n' + local + \
            '            KERNEL_LOCAL_VIRIAL\n        } else {\n' + loop + '        }\n'
        kernel = kernel[:i0] + block + kernel[i1:]
    return kernel


########################################################################
#                                                                      #
# variable                                                             #
#                                                                      #
########################################################################


class Variable:
    def __init__(self, iork:str, defstr:str) -> None:
        vs = defstr.split()
        self.iork = iork
        self.location = vs[0]  # shared / register
        self.type = vs[1]
        self.name = vs[2]
        self.readfrom, self.addto, self.onlyif = None, None, None
        self.scale, self.dlsink = None, None
        self.ifset = None
        # sorted: the source is in the order of the spatial list, read by the
        # sorted index of the atom (and through bnum in the exclusion loop)
        self.sorted = False
        for i in range(3, len(vs)):
            s2 = vs[i]
            if s2 == 'sorted':
                self.sorted = True
            elif s2.startswith('from:'):
                s3 = s2.replace('from:', '')
                self.readfrom = s3
            elif s2.startswith('addto:'):
                s3 = s2.replace('addto:', '')
                self.addto = s3
            elif s2.startswith('onlyif:'):
                s3 = s2.replace('onlyif:', '')
                self.onlyif = s3
            elif s2.startswith('ifset:'):
                # runtime guard: read 0 when this pointer is null
                self.ifset = s2.replace('ifset:', '')
            elif s2.startswith('scale:'):
                self.scale = s2.replace('scale:', '')
            elif s2.startswith('dlsink:'):
                self.dlsink = s2.replace('dlsink:', '').split(',')
                if len(self.dlsink) != 3:
                    raise ValueError('dlsink: needs destination,scale,gate')

    @staticmethod
    def _get_src(src, index) -> str:
        if ',' in src:
            vs = src.split(',')
            return '{}[{}][{}]'.format(vs[0], index, vs[1])
        else:
            return '{}[{}]'.format(src, index)

    def _load(self, index=None) -> str:
        rhs = self._get_src(self.readfrom, self.iork if index is None else index)
        if self.ifset is not None:
            rhs = '({} ? {} : 0)'.format(self.ifset, rhs)
        return rhs

    def zero(self) -> str:
        v1 = ''
        if self.location == 'shared':
            v1 = '{}[threadIdx.x] = 0;'.format(self.name)
        else:
            v1 = '{} = 0;'.format(self.name)
        return v1

    def _add_to(self, dst, scale) -> str:
        suffix = '[threadIdx.x]' if self.location == 'shared' else ''
        value = '{}{}'.format(self.name, suffix)
        if scale is not None:
            value = '{} * {}'.format(scale, value)
        if ',' not in dst:
            return 'atomic_add({}, {}, {});'.format(value, dst, self.iork)
        vs = dst.split(',')
        return 'atomic_add({}, &{}[{}][{}]);'.format(value, vs[0], self.iork, vs[1])

    def save(self) -> str:
        v1 = self._add_to(self.addto, self.scale)
        if self.dlsink is not None:
            dst, scale, gate = self.dlsink
            v1 = v1 + 'if CONSTEXPR ({}) {}'.format(gate, self._add_to(dst, scale))
        return v1

    def save_count(self) -> int:
        # the dlsink adds a second statement
        return 1 if self.dlsink is None else 2

    def init_exclude(self) -> str:
        rhs = self._load('bnum[{}]'.format(self.iork) if self.sorted else None)
        if self.location == 'shared':
            if self.iork == 'i':
                return ('{}[klane] = {};'.format(self.name, rhs))
            elif self.iork == 'k':
                return ('{}[threadIdx.x] = {};'.format(self.name, rhs))
        else:
            return ('{} = {};'.format(self.name, rhs))

    def init_block(self) -> str:
        if self.readfrom in ['x', 'y', 'z']:
            if self.location == 'shared':
                return ('{}[threadIdx.x] = sorted[atom{}].{};'.format(self.name, self.iork, self.readfrom))
            else:
                return ('{} = sorted[atom{}].{};'.format(self.name, self.iork, self.readfrom))
        else:
            rhs = self._load('atom' + self.iork if self.sorted else None)
            if self.location == 'shared':
                return ('{}[threadIdx.x] = {};'.format(self.name, rhs))
            else:
                return ('{} = {};'.format(self.name, rhs))

    def shuffle(self) -> str:
        if self.location == 'register':
            return ('{0:} = __shfl_sync(ALL_LANES, {0:}, ilane + 1);'.format(self.name))
        else:
            raise ValueError('Cannot shuffle variables in the shared memory.')

    def ikreplace(self, code:str) -> str:
        old_name = '@{}@'.format(self.name)
        new_name = self.name
        if self.iork == 'i':
            if self.location == 'shared':
                new_name = new_name + '[klane]'
        else:
            if self.location == 'shared':
                new_name = new_name + '[threadIdx.x]'
        code = code.replace(old_name, new_name)
        return code

    def iterreplace(self, code:str) -> str:
        old_name = '@{}@'.format('i')
        new_name = self.name
        code = code.replace(old_name, new_name)
        return code


class VariableDefinitions:
    def __init__(self, iork:str, lst:list) -> None:
        self.shared, self.register = {}, {}
        for defstr in lst:
            v = Variable(iork, defstr)
            loc = v.location
            if loc == 'shared':
                d = self.shared
            elif loc == 'register':
                d = self.register
            if v.type in d.keys():
                d[v.type].append(v)
            else:
                d[v.type] = [v]

    def declare(self) -> str:
        s = ''
        for t in self.shared.keys():
            s = s + ('__shared__ %s' % t)
            for v in self.shared[t]:
                s = s + (' %s[BLOCK_DIM],' % v.name)
            s = s + ';'
        for t in self.register.keys():
            s = s + ('%s' % t)
            for v in self.register[t]:
                s = s + (' %s,' % v.name)
            s = s + ';'
        s = s.replace(',;', ';')
        return s

    def _all(self, shared=True):
        vs = []
        if shared:
            for t in self.shared.keys():
                vs = vs + self.shared[t]
        for t in self.register.keys():
            vs = vs + self.register[t]
        return vs

    @staticmethod
    def _gated(items) -> str:
        # items are (onlyif, statements, count). onlyif: elides the load and the
        # shuffle along with zero and save; consecutive statements under the same
        # condition share one block.
        s = ''
        i = 0
        while i < len(items):
            cond = items[i][0]
            body, count = '', 0
            while i < len(items) and items[i][0] == cond:
                body = body + items[i][1]
                count = count + items[i][2]
                i = i + 1
            if cond is None:
                s = s + body
            elif count > 1:
                s = s + 'if CONSTEXPR ({}) {{{}}}'.format(cond, body)
            else:
                s = s + 'if CONSTEXPR ({}) {}'.format(cond, body)
        return s

    def zero(self) -> str:
        return self._gated([(v.onlyif, v.zero(), 1) for v in self._all()])

    def save(self) -> str:
        return self._gated([(v.onlyif, v.save(), v.save_count()) for v in self._all()])

    def init_exclude(self) -> str:
        return self._gated([(v.onlyif, v.init_exclude(), 1) for v in self._all()])

    def init_block(self) -> str:
        return self._gated([(v.onlyif, v.init_block(), 1) for v in self._all()])

    def shuffle(self) -> str:
        return self._gated([(v.onlyif, v.shuffle(), 1) for v in self._all(shared=False)])

    def any_sorted(self) -> bool:
        return any(v.sorted for v in self._all())

    def coordinates(self):
        # the variables read from x, y and z, in that order, as this thread
        # holds them, and whether any is in shared memory
        c = {v.readfrom: v for v in self._all() if v.readfrom in ['x', 'y', 'z']}
        if len(c) != 3:
            raise ValueError('LOCAL_TILES needs x, y and z.')
        own = lambda v: v.name + ('[threadIdx.x]' if v.location == 'shared' else '')
        return own(c['x']), own(c['y']), own(c['z']), any(v.location == 'shared' for v in c.values())

    def ikreplace(self, code:str) -> str:
        for t in self.shared.keys():
            for v in self.shared[t]:
                code = v.ikreplace(code)
        for t in self.register.keys():
            for v in self.register[t]:
                code = v.ikreplace(code)
        return code


########################################################################
#                                                                      #
# writer                                                               #
#                                                                      #
########################################################################


class KernelWriter:
    @staticmethod
    def _func_param(ptype:str, pname:str) -> str:
        if ptype in ['int', 'real']:
            return ', {} {}'.format(ptype, pname)
        elif ptype == 'int_const_array':
            return ', const int* restrict {}'.format(pname)
        elif ptype == 'real_const_array':
            return ', const real* restrict {}'.format(pname)
        elif 'int2_const_array' in ptype:
            return ', const int (*restrict {})[2]'.format(pname)
        elif 'real2_const_array' in ptype:
            return ', const real (*restrict {})[2]'.format(pname)
        elif 'real3_const_array' in ptype:
            return ', const real (*restrict {})[3]'.format(pname)
        elif 'real4_const_array' in ptype:
            return ', const real (*restrict {})[4]'.format(pname)
        else:
            raise ValueError('Do not know how to parse type: {}'.format(ptype))

    def _image_call(self, args:str) -> str:
        # IMAGE_TEMPLATE names the template class that images for the box shape;
        # the shapes without a class of their own image by the box_shape argument
        img = self._kv(self.yk_image_template)
        if img:
            return '{}::img2({}, TINKER_IMAGE_ARGS)'.format(img, args)
        return 'image2({})'.format(args)

    def _pair_body(self, code:str, local:bool) -> str:
        # image2(x, y, z) of a pair body images for the box shape, or, in a
        # local tile, is the plain distance squared
        def sub(m):
            args = m.group(1)
            if local:
                a = [t.strip() for t in args.split(',')]
                return '({0} * {0} + {1} * {1} + {2} * {2})'.format(*a)
            return self._image_call(args)
        code = re.sub(r'\bimage2\(([^()]*)\)', sub, code)
        # only bodies that read tile_local declare it
        if self._kv(self.yk_local_tiles) and re.search(r'\btile_local\b', code):
            code = 'constexpr bool tile_local = {};'.format('true' if local else 'false') + code
        return code

    @staticmethod
    def _load_scale_param(ptype:str, stem:str, input:str, separate_scaled_pairwise:bool) -> str:
        if ptype == 'real_const_array':
            v = ''
            if input is None:
                if not separate_scaled_pairwise:
                    v = 'real {}a = 1;'.format(stem)
            else:
                v = 'real {}a = {}[ii];'.format(stem, input)
            return v
        else:
            pattern = re.compile('(\\D*)(\\d*)_const_array,')
            match = re.match(pattern, ptype)
            if match and len(match.groups()) == 2:
                # real4_const_array ==> (real) (4)
                t = match.group(1)
                # dim = match.group(2)
                ss = ptype.split(',')
                v = ''
                for i in range(1, len(ss)):
                    idx = ss[i]
                    al = rc_alphabets[idx]
                    if input is None:
                        if not separate_scaled_pairwise:
                            v = v + '{} {}{} = 1;'.format(t, stem, al)
                    else:
                        v = v + '{} {}{} = {}[ii][{}];'.format(t, stem, al, input, idx)
                return v

    def __init__(self, config) -> None:
        self.config = config

        self.yk_kernel_version_number = 'KERNEL_VERSION_NUMBER'

        self.yk_output_dir = 'OUTPUT_DIR'
        self.yk_kernel_is_static = 'KERNEL_IS_STATIC'
        self.yk_kernel_name = 'KERNEL_NAME'
        self.yk_template_params = 'TEMPLATE_PARAMS'
        self.yk_constexpr_flags = 'CONSTEXPR_FLAGS'

        self.yk_cut_distance = 'CUT_DISTANCE'
        self.yk_off_distance = 'OFF_DISTANCE'
        self.yk_exclude_info = 'EXCLUDE_INFO'
        self.yk_scale_1x_type = 'SCALE_1X_TYPE'
        self.yk_extra_params = 'EXTRA_PARAMS'
        self.yk_implicit_params = 'IMPLICIT_PARAMS'

        self.yk_count, self.yk_energy = 'COUNT', 'ENERGY'
        self.yk_virial, self.yk_gradient = 'VIRIAL', 'GRADIENT'
        self.yk_i_variables, self.yk_k_variables = 'I_VARIABLES', 'K_VARIABLES'
        self.yk_i_force, self.yk_k_force = 'I_FORCE', 'K_FORCE'

        self.yk_scaled_pairwise = 'SCALED_PAIRWISE_INTERACTION'
        self.yk_full_pairwise = 'FULL_PAIRWISE_INTERACTION'

        self.yk_image_template = 'IMAGE_TEMPLATE'
        self.yk_local_tiles = 'LOCAL_TILES'
        self.yk_local_virial = 'LOCAL_VIRIAL'

        self.yk_single_loop_limit = 'SINGLE_LOOP_LIMIT'
        self.yk_single_loop_iter = 'SINGLE_LOOP_ITER'
        self.yk_single_loop_code = 'SINGLE_LOOP_CODE'

    def _kv(self, k:str):
        if k in self.config.keys():
            return self.config[k]
        else:
            return ''

    def cudaReplaceDict(self) -> dict:
        d = {}
        config = self.config
        keys = config.keys()

        # template
        k, v = 'TEMPLATE_PARAMS', self._kv(self.yk_template_params)
        d[k] = v

        # kernel name
        k, v = 'STATIC_KERNEL', ''
        kcfg, vcfg = self.yk_kernel_is_static, False
        if kcfg in keys:
            vcfg = config[kcfg]
        if vcfg:
            v = 'static'
        d[k] = v
        k, v = 'KERNEL_NAME', self._kv(self.yk_kernel_name)
        d[k] = v

        # cut, off
        k1, v1 = 'CUT_KERNEL_PARAMS', ''
        kcfg = self.yk_cut_distance
        if kcfg in keys:
            t = config[kcfg]
            v1 = v1 + ', real {}'.format(t)
        k2, v2 = 'OFF_KERNEL_PARAMS', ''
        kcfg = self.yk_off_distance
        if kcfg in keys:
            t = config[kcfg]
            v2 = v2 + ', real {}'.format(t)
        d[k1], d[k2] = v1, v2

        # extra kernel parameters
        k, v = 'EXTRA_KERNEL_PARAMS', self._kv(self.yk_extra_params)
        d[k] = v

        k, v = 'USING_DEVICE_VARIABLES', ''
        kcfg = self.yk_implicit_params
        if kcfg in keys:
            vcfg = config[kcfg]
            for t in vcfg:
                v = v + 'using {};'.format(t)
        d[k] = v

        k, v = 'KERNEL_CONSTEXPR_FLAGS', self._kv(self.yk_constexpr_flags)
        d[k] = v

        use_ikvars = False
        if self.yk_i_variables in keys and self.yk_k_variables in keys:
            use_ikvars = True
        if use_ikvars:
            # i and k declaration
            ivars, kvars = VariableDefinitions('i', config[self.yk_i_variables]), VariableDefinitions('k', config[self.yk_k_variables])
            ifrcs, kfrcs = VariableDefinitions('i', config[self.yk_i_force]), VariableDefinitions('k', config[self.yk_k_force])
            if len(ifrcs.shared.keys()):
                raise ValueError('I_FORCE cannot be put on shared memory.')
            if len(kfrcs.shared.keys()):
                raise ValueError('F_FORCE cannot be put on shared memory.')
            k1, v1 = 'DECLARE_PARAMS_I_AND_K', ivars.declare() + kvars.declare()
            ifdl = VariableDefinitions('i', config['I_FORCE_DLMDA']) if 'I_FORCE_DLMDA' in keys else None
            kfdl = VariableDefinitions('k', config['K_FORCE_DLMDA']) if 'K_FORCE_DLMDA' in keys else None
            k2, v2 = 'DECLARE_FORCE_I_AND_K', ifrcs.declare() + kfrcs.declare()
            if ifdl is not None:
                v2 = v2 + ifdl.declare() + kfdl.declare()
            d[k1], d[k2] = v1, v2

            # i and k in exclude block
            k1, v1 = 'KERNEL_INIT_EXCLUDE_PARAMS_I_AND_K', ''
            k2, v2 = 'KERNEL_INIT_PARAMS_I_AND_K', ''
            k3, v3 = 'KERNEL_SHUFFLE_PARAMS_I', ''
            v1 = v1 + ivars.init_exclude() + kvars.init_exclude()
            v2 = v2 + ivars.init_block() + kvars.init_block()
            v3 = v3 + ivars.shuffle()
            d[k1], d[k2], d[k3] = v1, v2, v3

        # count
        k1, v1 = 'COUNT_KERNEL_PARAMS', ''
        k2, v2 = 'DECLARE_ZERO_LOCAL_COUNT', ''
        k3, v3 = 'KERNEL_SUM_COUNT', ''
        kcfg = self.yk_count
        if kcfg in keys:
            vcfg, decl, zero, total = config[kcfg], '', '', ''
            for t in vcfg:
                v1 = v1 + ', CountBuffer restrict {}'.format(t)
                decl = decl + 'int {}tl;'.format(t)
                zero = zero + '{}tl = 0;'.format(t)
                total = total + 'atomic_add({}tl, {}, ithread);'.format(t, t)
            v2 = '%s if CONSTEXPR (do_a) {%s}' % (decl, zero)
            v3 = 'if CONSTEXPR (do_a) {%s}' % (total)
        d[k1], d[k2], d[k3] = v1, v2, v3

        # energy
        k1, v1 = 'ENERGY_KERNEL_PARAMS', ''
        k2, v2 = 'DECLARE_ZERO_LOCAL_ENERGY', ''
        k3, v3 = 'KERNEL_SUM_ENERGY', ''
        kcfg = self.yk_energy
        if kcfg in keys:
            vcfg, decl, zero, total = config[kcfg], '', '', ''
            for t in vcfg:
                v1 = v1 + ', EnergyBuffer restrict {}'.format(t)
                decl = decl + 'ebuf_prec {}tl;'.format(t)
                zero = zero + '{}tl = 0;'.format(t)
                total = total + 'atomic_add({}tl, {}, ithread);'.format(t, t)
            v2 = 'using ebuf_prec = EnergyBufferTraits::type;\n'
            v2 = v2 + '%s if CONSTEXPR (do_e) {%s}' % (decl, zero)
            v3 = 'if CONSTEXPR (do_e) {%s}' % total
            for kdl, gate in [('ENERGY_DLMDA1', 'do_dl1'), ('ENERGY_DLMDA2', 'do_dl2')]:
                if kdl not in keys:
                    continue
                decl, zero, total = '', '', ''
                for t in config[kdl]:
                    v1 = v1 + ', EnergyBuffer restrict {}'.format(t)
                    decl = decl + 'ebuf_prec {}tl;'.format(t)
                    zero = zero + '{}tl = 0;'.format(t)
                    total = total + 'atomic_add({}tl, {}, ithread);'.format(t, t)
                v2 = v2 + '%s if CONSTEXPR (do_e) {%s}' % (decl, zero)
                v3 = v3 + 'if CONSTEXPR (%s) {%s}' % (gate, total)
        d[k1], d[k2], d[k3] = v1, v2, v3

        # virial
        k1, v1 = 'VIRIAL_KERNEL_PARAMS', ''
        k2, v2 = 'DECLARE_ZERO_LOCAL_VIRIAL', ''
        k3, v3 = 'KERNEL_SUM_VIRIAL', ''
        kcfg = self.yk_virial
        if kcfg in keys:
            vcfg, decl, zero, total = config[kcfg], '', '', ''
            for t in vcfg:
                v1 = v1 + ', VirialBuffer restrict {}'.format(t)
                decl = decl + 'vbuf_prec {}tlxx'.format(t)
                zero = zero + '{}tlxx = 0;'.format(t)
                total = total + 'atomic_add({}tlxx'.format(t)
                for sufx in ['yx', 'zx', 'yy', 'zy', 'zz']:
                    decl = decl + ', {}tl{}'.format(t, sufx)
                    zero = zero + '{}tl{} = 0;'.format(t, sufx)
                    total = total + ', {}tl{}'.format(t, sufx)
                decl = decl + ';'
                zero = zero + '\n'
                total = total + ', {}, ithread);'.format(t)
            v2 = 'using vbuf_prec = VirialBufferTraits::type;\n'
            v2 = v2 + '%s if CONSTEXPR (do_v) {%s}' % (decl, zero)
            v3 = 'if CONSTEXPR (do_v) {%s}' % total
            if 'VIRIAL_DLMDA' in keys:
                decl, zero, total = '', '', ''
                for t in config['VIRIAL_DLMDA']:
                    v1 = v1 + ', VirialBuffer restrict {}'.format(t)
                    decl = decl + 'vbuf_prec {}tlxx'.format(t)
                    zero = zero + '{}tlxx = 0;'.format(t)
                    total = total + 'atomic_add({}tlxx'.format(t)
                    for sufx in ['yx', 'zx', 'yy', 'zy', 'zz']:
                        decl = decl + ', {}tl{}'.format(t, sufx)
                        zero = zero + '{}tl{} = 0;'.format(t, sufx)
                        total = total + ', {}tl{}'.format(t, sufx)
                    decl = decl + ';'
                    zero = zero + '\n'
                    total = total + ', {}, ithread);'.format(t)
                v2 = v2 + '%s if CONSTEXPR (do_v) {%s}' % (decl, zero)
                v3 = v3 + 'if CONSTEXPR (do_vdl) {%s}' % total
        d[k1], d[k2], d[k3] = v1, v2, v3

        if use_ikvars:
            # sync warp
            k1, v1 = 'KERNEL_SYNCWARP', '__syncwarp();'
            if len(ivars.shared) == 0 and len(kvars.shared) == 0 and len(ifrcs.shared) == 0 and len(kfrcs.shared) == 0:
                v1 = ''
            d[k1] = v1

            # gradient
            k, v = 'GRADIENT_KERNEL_PARAMS', ''
            kcfg = self.yk_gradient
            if kcfg in keys:
                vcfg = config[kcfg]
                for t in vcfg:
                    v = v + ', grad_prec* restrict {}'.format(t)
                if 'GRADIENT_DLMDA' in keys:
                    for t in config['GRADIENT_DLMDA']:
                        v = v + ', grad_prec* restrict {}'.format(t)
            k1, v1 = 'KERNEL_ZERO_LOCAL_FORCE', ifrcs.zero() + kfrcs.zero()
            k2, v2 = 'KERNEL_SAVE_LOCAL_FORCE', ifrcs.save() + kfrcs.save()
            k3, v3 = 'KERNEL_SHUFFLE_LOCAL_FORCE_I', ifrcs.shuffle()
            kcfg = self.yk_constexpr_flags
            if kcfg in keys:
                vcfg = config[kcfg]
                if 'constexpr bool do_g =' in vcfg:
                    v1 = 'if CONSTEXPR (do_g) {%s}' % v1
                    v2 = 'if CONSTEXPR (do_g) {%s}' % v2
                    if ifdl is not None:
                        v1 = v1 + 'if CONSTEXPR (do_gdl) {%s}' % (ifdl.zero() + kfdl.zero())
                        v2 = v2 + 'if CONSTEXPR (do_gdl) {%s}' % (ifdl.save() + kfdl.save())
                if v3 != '':
                    v3 = 'if CONSTEXPR (do_g) {%s}' % v3
                    if ifdl is not None:
                        v3 = v3 + 'if CONSTEXPR (do_gdl) {%s}' % ifdl.shuffle()
            d[k], d[k1], d[k2], d[k3] = v, v1, v2, v3

            # klane -- True only if ivar uses shared memory
            k1, v1 = 'KERNEL_KLANE1', ''
            k2, v2 = 'KERNEL_KLANE2', ''
            k3, v3 = 'KERNEL_SCALED_KLANE', ''
            use_klane = False
            if len(ivars.shared.keys()):
                use_klane = True
            if use_klane:
                v1 = 'int klane = srclane + threadIdx.x - ilane;'
                v2 = v2 + 'int srclane = (ilane + j) & (WARP_SIZE - 1);'
                v2 = v2 + 'int klane = srclane + threadIdx.x - ilane;'
                v3 = 'const int klane = threadIdx.x;'
            d[k1], d[k2], d[k3] = v1, v2, v3

        # exclude
        k1, v1 = 'EXCLUDE_INFO_KERNEL_PARAMS', ''
        k2, v2 = 'KERNEL_LOAD_INFO_VARIABLES', ''
        k3, v3 = 'KERNEL_EXCLUDE_BIT', ''
        kcfg = self.yk_exclude_info
        if kcfg in keys:
            t = config[kcfg]
            v1 = v1 + ', const unsigned* restrict {}'.format(t)
            v2 = v2 + 'unsigned int {0:}0 = {0:}[iw * WARP_SIZE + ilane];'.format(t)
            v3 = v3 + ' and ({}0 & srcmask) == 0'.format(t)
        if v3 != '':
            v3 = 'int srcmask = 1 << srclane;' + 'incl = incl {};'.format(v3)
        d[k1], d[k2], d[k3] = v1, v2, v3

        k0, v0 = 'EXCLUDE_SCALE_KERNEL_PARAMS', ''
        k1, v1 = 'KERNEL_LOAD_1X_SCALES', ''
        k2, v2 = 'KERNEL_HAS_1X_SCALE', '/* /'
        k3, v3 = 'KERNEL_SCALE_1', ''
        kcfg = self.yk_scale_1x_type
        separate_scaled_pairwise = self.yk_scaled_pairwise in keys
        if kcfg in keys:
            vcfg = config[kcfg]
            v0 = ', int nexclude, const int (*restrict exclude)[2]'
            v0 = v0 + self._func_param(vcfg, 'exclude_scale')
            v1 = self._load_scale_param(vcfg, 'scale', 'exclude_scale', separate_scaled_pairwise)
            v2 = '//* /'
            v3 = self._load_scale_param(vcfg, 'scale', None, separate_scaled_pairwise)
        v0 = v0 + ', const real* restrict x, const real* restrict y, const real* restrict z'
        d[k0], d[k1], d[k2], d[k3] = v0, v1, v2, v3

        # interaction
        k1, v1 = 'KERNEL_FULL_PAIRWISE_INTERACTION', ''
        k2, v2 = 'KERNEL_SCALED_PAIRWISE_INTERACTION', ''
        kcfg = self.yk_full_pairwise
        if kcfg in keys:
            v1 = config[kcfg]
            v1 = ivars.ikreplace(v1)
            v1 = kvars.ikreplace(v1)
            v1 = ifrcs.ikreplace(v1)
            v1 = kfrcs.ikreplace(v1)
            if ifdl is not None:
                v1 = ifdl.ikreplace(v1)
                v1 = kfdl.ikreplace(v1)
            v2 = v1  # in case no scaled pairwise interaction is given
        kcfg = self.yk_scaled_pairwise
        if kcfg in keys:
            v2 = config[kcfg]
            v2 = ivars.ikreplace(v2)
            v2 = kvars.ikreplace(v2)
            v2 = ifrcs.ikreplace(v2)
            v2 = kfrcs.ikreplace(v2)
            if ifdl is not None:
                v2 = ifdl.ikreplace(v2)
                v2 = kfdl.ikreplace(v2)
        k3, v3 = 'KERNEL_LOCAL_PAIRWISE_INTERACTION', ''
        if self._kv(self.yk_local_virial) not in ['', 'symmetric']:
            raise ValueError('LOCAL_VIRIAL takes symmetric or nothing.')
        if self._kv(self.yk_local_virial) and not self._kv(self.yk_local_tiles):
            raise ValueError('LOCAL_VIRIAL needs LOCAL_TILES.')
        if self._kv(self.yk_image_template) or self._kv(self.yk_local_tiles):
            if self._kv(self.yk_local_tiles):
                # a local tile takes its virial from the summed forces, so the
                # pair virials of the body must skip it
                if (self.yk_virial in keys or 'VIRIAL_DLMDA' in keys) and not re.search(r'\btile_local\b', v1):
                    raise ValueError('LOCAL_TILES with a virial needs the pair virials guarded by tile_local.')
                v3 = self._pair_body(v1, True)
            v1, v2 = self._pair_body(v1, False), self._pair_body(v2, False)
        d[k1], d[k2], d[k3] = v1, v2, v3

        # sorted sources and local tiles
        k1, v1 = 'SORTED_KERNEL_PARAMS', ''
        if use_ikvars and (ivars.any_sorted() or kvars.any_sorted()):
            # only kernel 21 takes bnum for its exclusion loop
            if self.config.get(self.yk_kernel_version_number, 21) != 21:
                raise ValueError('sorted sources need KERNEL_VERSION_NUMBER 21.')
            v1 = ', const int* restrict bnum'
        k2, v2 = 'LOCAL_KERNEL_PARAMS', ''
        k3, v3 = 'KERNEL_LOCAL_IMAGE', ''
        k4, v4 = 'KERNEL_LOCAL_VIRIAL', ''
        k5, v5 = 'KERNEL_LOCAL_FLAG', ''
        if self._kv(self.yk_local_tiles):
            v2 = ', const Spatial::Center* restrict akc'
            v5 = 'real xc = akc[ty].x; real yc = akc[ty].y; real zc = akc[ty].z;'
            v5 = v5 + 'const bool ilocal = akc[ty].w != 0;'
            # the coordinates stay relative to the center: the pair bodies take
            # only their differences, and the forces of a tile sum to zero, so
            # its virial is the same and keeps the round-off of a small box
            v3 = 'real xr, yr, zr;'
            shared = False
            for vs in [ivars, kvars]:
                x, y, z, sh = vs.coordinates()
                shared = shared or sh
                v3 = v3 + 'xr = {} - xc; yr = {} - yc; zr = {} - zc;'.format(x, y, z)
                v3 = v3 + self._image_call('xr, yr, zr') + ';'
                v3 = v3 + '{} = xr; {} = yr; {} = zr;'.format(x, y, z)
            if shared:
                # the other lanes read the shared coordinates
                v3 = v3 + '__syncwarp();'
            xi, yi, zi, _ = ivars.coordinates()
            xk, yk, zk, _ = kvars.coordinates()
            for kv, kfi, kfk, gate in [(self.yk_virial, self.yk_i_force, self.yk_k_force, 'do_v'),
                                       ('VIRIAL_DLMDA', 'I_FORCE_DLMDA', 'K_FORCE_DLMDA', 'do_vdl')]:
                if kv not in keys:
                    continue
                if len(config[kv]) != 1:
                    raise ValueError('LOCAL_TILES takes one virial buffer each.')
                t = config[kv][0]
                if kfi in keys:
                    fiv = VariableDefinitions('i', config[kfi])._all()[0:3]
                    fkv = VariableDefinitions('k', config[kfk])._all()[0:3]
                    scales = [v.scale for v in fiv + fkv]
                else:
                    # the forces sink their lambda derivatives with a second
                    # scale (dlsink)
                    fiv = VariableDefinitions('i', config[self.yk_i_force])._all()[0:3]
                    fkv = VariableDefinitions('k', config[self.yk_k_force])._all()[0:3]
                    scales = [v.dlsink[1] if v.dlsink else None for v in fiv + fkv]
                    if scales[0] is None:
                        raise ValueError('LOCAL_TILES needs {} or dlsink on the forces.'.format(kfi))
                if len(set(scales)) != 1:
                    raise ValueError('LOCAL_TILES needs one scale on the forces.')
                # the first three forces are the gradient along x, y and z: their
                # arrays differ in one letter, x, y and z
                for fv in [fiv, fkv]:
                    dst = [v.dlsink[0] if kfi not in keys else v.addto for v in fv]
                    diff = [i for i in range(len(dst[0])) if len(set(d[i:i + 1] for d in dst)) != 1]
                    if any(d is None or len(d) != len(dst[0]) for d in dst) or len(diff) != 1 or \
                       [d[diff[0]] for d in dst] != ['x', 'y', 'z']:
                        raise ValueError('LOCAL_TILES needs the x, y and z gradient as the first three forces.')
                fi = [v.name for v in fiv]
                fk = [v.name for v in fkv]
                pos_i, pos_k = [xi, yi, zi], [xk, yk, zk]
                # the pair virial of a force off the line of centers, as of
                # multipoles, is the symmetric part of r times f
                symmetric = self._kv(self.yk_local_virial) == 'symmetric'
                body = ''
                for sufx, a, b in [('xx', 0, 0), ('yx', 1, 0), ('zx', 2, 0), ('yy', 1, 1), ('zy', 2, 1), ('zz', 2, 2)]:
                    term = '{} * {} + {} * {}'.format(pos_i[a], fi[b], pos_k[a], fk[b])
                    if symmetric and a != b:
                        term = '0.5f * ({} + {} * {} + {} * {})'.format(term, pos_i[b], fi[a], pos_k[b], fk[a])
                    if scales[0] is not None:
                        term = '{} * ({})'.format(scales[0], term)
                    body = body + '{}tl{} += floatTo<vbuf_prec>({});'.format(t, sufx, term)
                v4 = v4 + 'if CONSTEXPR ({}) {{{}}}'.format(gate, body)
        d[k1], d[k2], d[k3], d[k4], d[k5] = v1, v2, v3, v4, v5

        # single loop
        k0, v0 = 'SINGLE_LOOP_LIMIT_PARAM', ''
        k1, v1 = 'KERNEL_SINGLE_LOOP_CODE', ''
        k2, v2 = 'KERNEL_SINGLE_LOOP_BEGIN', ''
        k3, v3 = 'KERNEL_SINGLE_LOOP_END', ''
        kcfg = self.yk_single_loop_code
        if kcfg in keys:
            v0 = config[self.yk_single_loop_limit]
            v1 = config[kcfg]
            sl_limit, sl_iter = config[self.yk_single_loop_limit], config[self.yk_single_loop_iter]
            sl_limit = 'register ' + sl_limit + ' from:dummy'
            sl_iter = 'register ' + sl_iter + ' from:dummy'
            var_limit = Variable('k', sl_limit)
            var_iter = Variable('k', sl_iter)
            v2 = 'for(%s %s = ITHREAD; %s < %s; %s += STRIDE) {' % (var_iter.type, var_iter.name, var_iter.name, var_limit.name, var_iter.name)
            v3 = '}'
            v1 = var_iter.iterreplace(v1)
        d[k0], d[k1], d[k2], d[k3] = v0, v1, v2, v3

        return d

    @staticmethod
    def version() -> str:
        return '3.2.0'

    @staticmethod
    def _replace(s:str, d:dict) -> str:
        output = s
        for k in d.keys():
            v = d[k]
            if v is None:
                v = ''
            output = output.replace(k, v)
        return output

    def write(self, output) -> None:
        d = self.cudaReplaceDict()
        outstr = '// ck.py Version {}'.format(self.version())
        kernel_num = 21  # default
        if self.yk_kernel_version_number in self.config.keys():
            kernel_num = self.config[self.yk_kernel_version_number]
        if kernel_num == 11:
            outstr = outstr + self._replace(rc_kernel11, d)
        else:
            # local tiles rewrite a template, given its number of tile loops
            local = _local_tiles if self._kv(self.yk_local_tiles) else (lambda k, nloop: k)
            if kernel_num == 23:
                if self.yk_scale_1x_type in self.config.keys():
                    outstr = outstr + self._replace(rc_kernel23c, d)
                outstr = outstr + self._replace(local(rc_kernel23b, 1), d)
                outstr = outstr + self._replace(local(rc_kernel23a, 1), d)
            else:
                outstr = outstr + self._replace(local(rc_kernel21, 2), d)
        print(outstr, file=output)


########################################################################
#                                                                      #
# main                                                                 #
#                                                                      #
########################################################################


def exec_mainfunc(argv):
    yaml_file = argv[1]
    with open(yaml_file) as input_file:
        config = yaml.full_load(input_file)
        kw = KernelWriter(config)
        kw.write(output=sys.stdout)


def show_command(argv):
    p = os.path.abspath(__file__)
    d = os.path.dirname(p)
    d2 = os.path.join(d, '../..')
    d = os.path.abspath(d2)

    yaml_file = argv[1]
    with open(yaml_file) as input_file:
        config = yaml.full_load(input_file)
        kw = KernelWriter(config)
        outd = config[kw.yk_output_dir]
        outf = config[kw.yk_kernel_name]
        print('python3 {} {} | clang-format | tee {}/{}/{}.cc'.format(__file__, argv[1], d, outd, outf))


if __name__ == '__main__':
    if sys.argv[1] == '-c':
        show_command(sys.argv[1:])
    else:
        exec_mainfunc(sys.argv)
