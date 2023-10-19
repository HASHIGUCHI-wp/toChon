// Gmsh project created on Sat Sep  9 10:53:55 2023
SetFactory("OpenCASCADE");

num_e = DefineNumber[1, Name "Parameters/num_e"];
element_size = DefineNumber[1.0, Name "Parameters/element_size"];
T = DefineNumber[num_e * element_size, Name "Parameters/T"];

alpha_L = DefineNumber[112, Name "Parameters/alpha_L"];
alpha_H = DefineNumber[19, Name "Parameters/alpha_H"];
alpha_W = DefineNumber[15, Name "Parameters/alpha_W"];

alpha_Lc = DefineNumber[6, Name "Parameters/alpha_Lc"]; // 空洞内部の長さ
alpha_Lt = DefineNumber[1, Name "Parameters/alpha_Lt"]; // 厚み x方向
alpha_Lte = DefineNumber[3, Name "Parameters/alpha_Lte"]; // 厚み x方向
alpha_Lj = DefineNumber[2, Name "Parameters/alpha_Lj"]; // 繋ぎ目 x方向

alpha_Wt = DefineNumber[2, Name "Parameters/alpha_Wt"]; // 厚み y方向
alpha_Wc = DefineNumber[11, Name "Parameters/alpha_Wc"]; // 空洞内部の幅
alpha_Wc1 = DefineNumber[4, Name "Parameters/alpha_Wc1"];
alpha_Wp = DefineNumber[alpha_Wc - 2 * alpha_Wc1, Name "Parameters/alpha_Wp"]; // 流路の幅

alpha_Hf = DefineNumber[4, Name "Parameters/alpha_Hf"]; // 基盤の高さ
alpha_Hp = DefineNumber[alpha_Wp, Name "Parameters/alpha_Hp"]; // 流路の高さ
alpha_Hj = DefineNumber[alpha_Hp + 1 , Name "Parameters/alpha_Hj"]; // 繋ぎ目の高さ
alpha_Hc = DefineNumber[13 , Name "Parameters/alpha_Hc"]; // 空洞の高さ
alpha_Hc1 = DefineNumber[alpha_Hc - alpha_Hj , Name "Parameters/alpha_Hc1"];
alpha_Ht = DefineNumber[2 , Name "Parameters/alpha_Ht"];


L = DefineNumber[alpha_L * T, Name "Parameters/L"];
H = DefineNumber[alpha_H * T, Name "Parameters/H"];
W = DefineNumber[alpha_W * T, Name "Parameters/W"];

Lc = DefineNumber[alpha_Lc * T, Name "Parameters/Lc"]; // 空洞内部の長さ
Lt = DefineNumber[alpha_Lt * T, Name "Parameters/Lt"]; // 厚み x方向
Lte = DefineNumber[alpha_Lte * T, Name "Parameters/Lte"]; // 厚み x方向
Lj = DefineNumber[alpha_Lj * T, Name "Parameters/Lj"]; // 繋ぎ目 x方向

Wt = DefineNumber[alpha_Wt * T, Name "Parameters/Wt"]; // 厚み y方向
Wc = DefineNumber[alpha_Wc * T, Name "Parameters/Wc"]; // 空洞内部の幅
Wc1 = DefineNumber[alpha_Wc1 * T, Name "Parameters/Wc1"];
Wp = DefineNumber[alpha_Wp * T, Name "Parameters/Wp"]; // 流路の幅

Hf = DefineNumber[alpha_Hf * T, Name "Parameters/Hf"]; // 基盤の高さ
Hp = DefineNumber[alpha_Hp * T, Name "Parameters/Hp"]; // 流路の高さ
Hj = DefineNumber[alpha_Hj * T, Name "Parameters/Hj"]; // 繋ぎ目の高さ
Hc = DefineNumber[alpha_Hc * T, Name "Parameters/Hc"]; // 空洞の高さ
Hc1 = DefineNumber[alpha_Hc1 * T, Name "Parameters/Hc1"];
Ht = DefineNumber[alpha_Ht * T, Name "Parameters/Ht"];


// 空洞部分
Box(1) = {Lte, 0.0, 0.0, Lc, Wt, Hf};
Box(2) = {Lte, 0.0, Hf, Lc, Wt, Hp};
Box(3) = {Lte, 0.0, Hf + Hp, Lc, Wt, Hj - Hp};
Box(4) = {Lte, 0.0, Hf + Hj, Lc, Wt, Hc1};
Box(5) = {Lte, 0.0, Hf + Hc, Lc, Wt, Ht};

Box(6) = {Lte, Wt, 0.0, Lc, Wc1, Hf};
// Box() = {Lte, Wt, Hf, Lc, Wc1, Hp};
// Box() = {Lte, Wt, Hf + Hp, Lc, Wc1, Hj - Hp};
// Box() = {Lte, Wt, Hf + Hj, Lc, Wc1, Hc1};
Box(7) = {Lte, Wt, Hf + Hc, Lc, Wc1, Ht};

Box(8) = {Lte, Wt + Wc1, 0.0, Lc, Wp, Hf};
// Box() = {Lte, Wt + Wc1, Hf, Lc, Wp, Hp};
// Box() = {Lte, Wt + Wc1, Hf + Hp, Lc, Wp, Hj - Hp};
// Box() = {Lte, Wt + Wc1, Hf + Hj, Lc, Wp, Hc1};
Box(9) = {Lte, Wt + Wc1, Hf + Hc, Lc, Wp, Ht};

Box(10) = {Lte, Wt + Wc1 + Wp, 0.0, Lc, Wc1, Hf};
// Box() = {Lte, Wt + Wc1 + Wp, Hf, Lc, Wc1, Hp};
// Box() = {Lte, Wt + Wc1 + Wp, Hf + Hp, Lc, Wc1, Hj - Hp};
// Box() = {Lte, Wt + Wc1 + Wp, Hf + Hj, Lc, Wc1, Hc1};
Box(11) = {Lte, Wt + Wc1 + Wp, Hf + Hc, Lc, Wc1, Ht};

Box(12) = {Lte, Wt + Wc1 + Wp + Wc1, 0.0, Lc, Wt, Hf};
Box(13) = {Lte, Wt + Wc1 + Wp + Wc1, Hf, Lc, Wt, Hp};
Box(14) = {Lte, Wt + Wc1 + Wp + Wc1, Hf + Hp, Lc, Wt, Hj - Hp};
Box(15) = {Lte, Wt + Wc1 + Wp + Wc1, Hf + Hj, Lc, Wt, Hc1};
Box(16) = {Lte, Wt + Wc1 + Wp + Wc1, Hf + Hc, Lc, Wt, Ht};


// 厚み部分
Box(17) = {Lte + Lc, 0.0, 0.0, Lt, Wt, Hf};
Box(18) = {Lte + Lc, 0.0, Hf, Lt, Wt, Hp};
Box(19) = {Lte + Lc, 0.0, Hf + Hp, Lt, Wt, Hj - Hp};
Box(20) = {Lte + Lc, 0.0, Hf + Hj, Lt, Wt, Hc1};
Box(21) = {Lte + Lc, 0.0, Hf + Hc, Lt, Wt, Ht};

Box(22) = {Lte + Lc, Wt, 0.0, Lt, Wc1, Hf};
Box(23) = {Lte + Lc, Wt, Hf, Lt, Wc1, Hp};
Box(24) = {Lte + Lc, Wt, Hf + Hp, Lt, Wc1, Hj - Hp};
Box(25) = {Lte + Lc, Wt, Hf + Hj, Lt, Wc1, Hc1};
Box(26) = {Lte + Lc, Wt, Hf + Hc, Lt, Wc1, Ht};

Box(27) = {Lte + Lc, Wt + Wc1, 0.0, Lt, Wp, Hf};
// Box() = {Lte + Lc, Wt + Wc1, Hf, Lt, Wp, Hp};
Box(28) = {Lte + Lc, Wt + Wc1, Hf + Hp, Lt, Wp, Hj - Hp};
Box(29) = {Lte + Lc, Wt + Wc1, Hf + Hj, Lt, Wp, Hc1};
Box(30) = {Lte + Lc, Wt + Wc1, Hf + Hc, Lt, Wp, Ht};

Box(31) = {Lte + Lc, Wt + Wc1 + Wp, 0.0, Lt, Wc1, Hf};
Box(32) = {Lte + Lc, Wt + Wc1 + Wp, Hf, Lt, Wc1, Hp};
Box(33) = {Lte + Lc, Wt + Wc1 + Wp, Hf + Hp, Lt, Wc1, Hj - Hp};
Box(34) = {Lte + Lc, Wt + Wc1 + Wp, Hf + Hj, Lt, Wc1, Hc1};
Box(35) = {Lte + Lc, Wt + Wc1 + Wp, Hf + Hc, Lt, Wc1, Ht};

Box(36) = {Lte + Lc, Wt + Wc1 + Wp + Wc1, 0.0, Lt, Wt, Hf};
Box(37) = {Lte + Lc, Wt + Wc1 + Wp + Wc1, Hf, Lt, Wt, Hp};
Box(38) = {Lte + Lc, Wt + Wc1 + Wp + Wc1, Hf + Hp, Lt, Wt, Hj - Hp};
Box(39) = {Lte + Lc, Wt + Wc1 + Wp + Wc1, Hf + Hj, Lt, Wt, Hc1};
Box(40) = {Lte + Lc, Wt + Wc1 + Wp + Wc1, Hf + Hc, Lt, Wt, Ht};


// 厚み部分
Box(41) = {Lte + Lc + Lt, 0.0, 0.0, Lj, Wt, Hf};
Box(42) = {Lte + Lc + Lt, 0.0, Hf, Lj, Wt, Hp};
Box(43) = {Lte + Lc + Lt, 0.0, Hf + Hp, Lj, Wt, Hj - Hp};
// Box() = {Lte + Lc + Lt, 0.0, Hf + Hj, Lj, Wt, Hc1};
// Box() = {Lte + Lc + Lt, 0.0, Hf + Hc, Lj, Wt, Ht};

Box(44) = {Lte + Lc + Lt, Wt, 0.0, Lj, Wc1, Hf};
Box(45) = {Lte + Lc + Lt, Wt, Hf, Lj, Wc1, Hp};
Box(46) = {Lte + Lc + Lt, Wt, Hf + Hp, Lj, Wc1, Hj - Hp};
// Box() = {Lte + Lc + Lt, Wt, Hf + Hj, Lj, Wc1, Hc1};
// Box() = {Lte + Lc + Lt, Wt, Hf + Hc, Lj, Wc1, Ht};

Box(47) = {Lte + Lc + Lt, Wt + Wc1, 0.0, Lj, Wp, Hf};
// Box() = {Lte + Lc + Lt, Wt + Wc1, Hf, Lj, Wp, Hp};
Box(48) = {Lte + Lc + Lt, Wt + Wc1, Hf + Hp, Lj, Wp, Hj - Hp};
// Box() = {Lte + Lc + Lt, Wt + Wc1, Hf + Hj, Lj, Wp, Hc1};
// Box() = {Lte + Lc + Lt, Wt + Wc1, Hf + Hc, Lj, Wp, Ht};

Box(49) = {Lte + Lc + Lt, Wt + Wc1 + Wp, 0.0, Lj, Wc1, Hf};
Box(50) = {Lte + Lc + Lt, Wt + Wc1 + Wp, Hf, Lj, Wc1, Hp};
Box(51) = {Lte + Lc + Lt, Wt + Wc1 + Wp, Hf + Hp, Lj, Wc1, Hj - Hp};
// Box() = {Lte + Lc + Lt, Wt + Wc1 + Wp, Hf + Hj, Lj, Wc1, Hc1};
// Box() = {Lte + Lc + Lt, Wt + Wc1 + Wp, Hf + Hc, Lj, Wc1, Ht};

Box(52) = {Lte + Lc + Lt, Wt + Wc1 + Wp + Wc1, 0.0, Lj, Wt, Hf};
Box(53) = {Lte + Lc + Lt, Wt + Wc1 + Wp + Wc1, Hf, Lj, Wt, Hp};
Box(54) = {Lte + Lc + Lt, Wt + Wc1 + Wp + Wc1, Hf + Hp, Lj, Wt, Hj - Hp};
// Box() = {Lte + Lc + Lt, Wt + Wc1 + Wp + Wc1, Hf + Hj, Lj, Wt, Hc1};
// Box() = {Lte + Lc + Lt, Wt + Wc1 + Wp + Wc1, Hf + Hc, Lj, Wt, Ht};



Translate {Lt + Lj, 0, 0} {
  Duplicata { Volume{17 : 40}; }
}

// 端部分
Box(79) = {0, 0.0, 0.0, Lte, Wt, Hf};
Box(80) = {0, 0.0, Hf, Lte, Wt, Hp};
Box(81) = {0, 0.0, Hf + Hp, Lte, Wt, Hj - Hp};
Box(82) = {0, 0.0, Hf + Hj, Lte, Wt, Hc1};
Box(83) = {0, 0.0, Hf + Hc, Lte, Wt, Ht};

Box(84) = {0, Wt, 0.0, Lte, Wc1, Hf};
Box(85) = {0, Wt, Hf, Lte, Wc1, Hp};
Box(86) = {0, Wt, Hf + Hp, Lte, Wc1, Hj - Hp};
Box(87) = {0, Wt, Hf + Hj, Lte, Wc1, Hc1};
Box(88) = {0, Wt, Hf + Hc, Lte, Wc1, Ht};

Box(89) = {0, Wt + Wc1, 0.0, Lte, Wp, Hf};
Box(90) = {0, Wt + Wc1, Hf, Lte, Wp, Hp};
Box(91) = {0, Wt + Wc1, Hf + Hp, Lte, Wp, Hj - Hp};
Box(92) = {0, Wt + Wc1, Hf + Hj, Lte, Wp, Hc1};
Box(93) = {0, Wt + Wc1, Hf + Hc, Lte, Wp, Ht};

Box(94) = {0, Wt + Wc1 + Wp, 0.0, Lte, Wc1, Hf};
Box(95) = {0, Wt + Wc1 + Wp, Hf, Lte, Wc1, Hp};
Box(96) = {0, Wt + Wc1 + Wp, Hf + Hp, Lte, Wc1, Hj - Hp};
Box(97) = {0, Wt + Wc1 + Wp, Hf + Hj, Lte, Wc1, Hc1};
Box(98) = {0, Wt + Wc1 + Wp, Hf + Hc, Lte, Wc1, Ht};

Box(99) = {0, Wt + Wc1 + Wp + Wc1, 0.0, Lte, Wt, Hf};
Box(100) = {0, Wt + Wc1 + Wp + Wc1, Hf, Lte, Wt, Hp};
Box(101) = {0, Wt + Wc1 + Wp + Wc1, Hf + Hp, Lte, Wt, Hj - Hp};
Box(102) = {0, Wt + Wc1 + Wp + Wc1, Hf + Hj, Lte, Wt, Hc1};
Box(103) = {0, Wt + Wc1 + Wp + Wc1, Hf + Hc, Lte, Wt, Ht};


Translate {1 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {2 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {3 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {4 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {5 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {6 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {7 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {8 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {9 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }
Translate {10 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{1 : 16}; } }


Translate {1 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }
Translate {2 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }
Translate {3 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }
Translate {4 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }
Translate {5 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }
Translate {6 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }
Translate {7 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }
Translate {8 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }
Translate {9 * (Lc + 2 * Lt + Lj), 0, 0} { Duplicata { Volume{17 : 78}; } }

Translate {10 * (Lc + 2 * Lt + Lj) + Lte + Lc, 0, 0} { Duplicata { Volume{79 : 103}; } }



BooleanFragments{ Volume{1}; Delete; }{ Volume{2:846}; Delete; }//+
Translate {3, 0, 0} {
  Duplicata { Surface{1162}; Curve{1219}; }
}
