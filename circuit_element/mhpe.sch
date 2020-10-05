W 0 1; down=.5
W 1 2; right=.5
W 2 3; up=.5
CPE 3 4; right, l_=dCPE
W 4 5; down=.5
W 5 9; right=.5
W 2 6; down=.5
Rct 6 7; right, l^=dR_{ct}
ZW 7 8; right, l^=dZ_W
W 8 5; up=.5
W 9 10; up=.5
RElectrolyte 9 12; down, l^=dR_{Electrolyte}
W 12 13; down=.5, dashed
RSolid 1 14; down, l_=dR_{Solid}
W 14 15; down=.5, dashed
W 16 0; down=.5, dashed
W 17 10; down=.5, dashed
; draw_nodes=none, label_nodes=none
;;\node[red,draw,dashed,inner sep=5mm,anchor=west, fit=(2) (8) (3), label=dRandles Circuit] {};
