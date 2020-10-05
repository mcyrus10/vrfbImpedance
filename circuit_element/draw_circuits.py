#!/Users/cyrus/miniconda3/bin/python3
from lcapy import Circuit
cct = Circuit("z_a.sch")
cct.draw('z_a.png')
cct = Circuit("z_b.sch")
cct.draw('z_b.png')
cct = Circuit('randles_circuit.sch')
cct.draw('z_randles.png')
cct = Circuit('mhpe.sch')
cct.draw('mhpe.png')
cct = Circuit('symmetric_cell.sch')
cct.draw('symmetric_cell.png')
