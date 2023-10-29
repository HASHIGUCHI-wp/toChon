// Gmsh project created on Sat Oct 28 13:43:05 2023
SetFactory("OpenCASCADE");


dx_mesh = DefineNumber[0.25, Name "Parameters/dx_mesh"];
L = DefineNumber[10.0, Name "Parameters/L"];
W = DefineNumber[1.0, Name "Parameters/W"];
H = DefineNumber[1.0, Name "Parameters/H"];

Box(1) = {0.0, 0.0, 0.0, L, W, H};

Transfinite Curve {9, 10, 11, 12} = Ceil(L / dx_mesh + 0.5) Using Progression 1;

Transfinite Curve {2, 6, 4, 8} = Ceil(W / dx_mesh + 0.5) Using Progression 1;

Transfinite Curve {3, 7, 1, 5} = Ceil(H / dx_mesh + 0.5) Using Progression 1;

Transfinite Surface "*";
Transfinite Volume "*";
