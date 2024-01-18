# This file is part of Qibolab
import cffi
ffibuilder = cffi.FFI()

with open('qibolab/qibolab.h') as f:
    ffibuilder.embedding_api(f.read())

ffibuilder.set_source('cqibolab', r'''
    #include "qibolab/qibolab.h"
''', source_extension='.c')

with open('wrapper.py') as f:
    ffibuilder.embedding_init_code(f.read())

ffibuilder.emit_c_code('cqibolab.c')
