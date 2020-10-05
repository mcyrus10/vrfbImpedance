W 0 1; right=.5
W 1 2; up=.5
W 2 3; left=.5
CPE 3 4; up, l_=dCPE
W 4 5; right=.5
W 5 9; up=1
W 2 6; right=.5
Rct 6 7; up, l^=dR_{ct}
ZW 7 8; up, l^=dZ_W
W 8 5; left=.5
W 9 10; left=.5
RElectrolyte 9 12; right, l^=dR_{Electrolyte}
W 12 13; right=.5, dashed
RSolid 1 14; right, l_=dR_{Solid}
W 14 15; right=.5, dashed
W 16 0; right=.5, dashed
W 17 10; right=.5, dashed
; draw_nodes=none, label_nodes=none
;;\node[red,draw,dashed,inner sep=5mm,anchor=west, fit=(2) (8) (3), label=dRandles Circuit] {};
